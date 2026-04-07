#!/usr/bin/env python3
"""
scanner.py

Authorized IoT testbed scanner for research environments.

Features:
- Modern Python 3.14 style
- Single-file design
- Pydantic validation
- Dataclasses for scan profiles
- Conservative staged Nmap scanning for fragile IoT devices
- XML parsing into JSON summaries
- Optional dry-run mode
- Optional sudo usage

Example:
    python scanner.py \
        --targets 192.168.1.10 192.168.1.15 \
        --output-dir scans \
        --stages 1 2 3

CIDR example:
    python scanner.py \
        --targets 192.168.1.0/24 \
        --stages 1 2 \
        --dry-run
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import shutil
import subprocess
import sys
import xml.etree.ElementTree as et

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

TargetString = Annotated[str, Field(min_length=1, max_length=128)]

AGGRESSIVE_SAFE_SCRIPTS: list[str] = [
    'dns-service-discovery',
    'broadcast-dns-service-discovery',
    'broadcast-bonjour',
    'broadcast-upnp-info',
    'upnp-info',
    'coap-resources',
    'snmp-info',
    'snmp-sysdescr',
    'snmp-interfaces',
    'ssl-cert',
    'ssl-enum-ciphers',
    'http-title',
    'http-headers',
    'http-enum',
    'banner',
    'rpcinfo',
    'nbstat',
    'ssh-auth-methods',
    'ssh-hostkey',
]

AGGRESSIVE_SCRIPT_ARGS: dict[str, str] = {
    'mqtt-subscribe.topic': '#',
    'mqtt-subscribe.listen-time': '30s',
    'mqtt-subscribe.listen-msgs': '100',
    'rtsp-url-brute.urlfile': '/usr/share/nmap/nselib/data/rtsp-urls.txt',
    'snmp-brute.communitiesdb': ('/usr/share/seclists/Discovery/SNMP/common-snmp-community-strings.txt'),
    'http-default-accounts.fingerprintfile': ('/usr/share/nmap/nselib/data/http-default-accounts-fingerprints.lua'),
}


class CliConfig(BaseModel):
    """Validated command-line configuration."""

    targets: list[TargetString]
    output_dir: Path = Field(default=Path('nmap_scans'))
    stages: list[Literal[1, 2, 3]] = Field(default_factory=lambda: [1, 2, 3])
    nmap_bin: str = Field(default='nmap')
    sudo: bool = Field(default=False)
    dry_run: bool = Field(default=False)
    stats_every: str = Field(default='30s')
    host_timeout: str = Field(default='10m')
    only_open: bool = Field(default=True)
    skip_ping: bool = Field(default=True)
    max_targets_per_run: int = Field(default=256, ge=1, le=4096)
    exclude: list[TargetString] = Field(default_factory=list)
    aggressive: bool = Field(default=False)
    script_args: dict[str, str] = Field(default_factory=dict)

    @field_validator('targets')
    @classmethod
    def validate_targets(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError('At least one target or CIDR must be provided.')

        for value in values:
            try:
                if '/' in value:
                    ipaddress.ip_network(value, strict=False)
                else:
                    ipaddress.ip_address(value)
            except ValueError as exc:
                raise ValueError(f'Invalid IP/CIDR target: {value}') from exc

        return values

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@dataclass(slots=True, frozen=True)
class ScanStage:
    """Represents one Nmap stage."""

    stage_id: int
    name: str
    description: str
    args: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ScanProfile:
    """Holds staged scan settings for an IoT-safe profile."""

    name: str
    stages: tuple[ScanStage, ...] = field(default_factory=tuple)


SAFE_IOT_PROFILE = ScanProfile(
    name='safe-iot-research',
    stages=(
        ScanStage(
            stage_id=1,
            name='initial_probe',
            description=(
                'Conservative TCP triage for common IoT service ports. Avoids aggressive timing for fragile devices.'
            ),
            args=(
                '-n',
                '-sS',
                '-T2',
                '--max-retries',
                '1',
                '--min-rate',
                '25',
                '--max-rate',
                '100',
                '-p',
                '21,22,23,53,80,443,554,1883,5000,5683,7547,8080,8443,8883,9999,37777,49152',
                '-sV',
                '--version-intensity',
                '3',
            ),
        ),
        ScanStage(
            stage_id=2,
            name='extended_enumeration',
            description=(
                'Broader TCP/UDP enumeration for common IoT and embedded services, still using polite timing.'
            ),
            args=(
                '-n',
                '-sS',
                '-sU',
                '-T2',
                '--max-retries',
                '2',
                '-p',
                (
                    'T:21,22,23,80,443,554,1883,5000,5683,7547,8080,8443,'
                    '8883,9999,18884,37777,49152-49157,'
                    'U:53,67,68,69,123,161,1900,3671,5353,5683'
                ),
                '-sV',
                '--version-intensity',
                '4',
            ),
        ),
        ScanStage(
            stage_id=3,
            name='metadata_and_service_scripts',
            description=(
                'NSE-based metadata and service discovery. '
                'Conservative by default; use --aggressive for brute-force, DoS, and intrusive scripts.'
            ),
            args=(
                '-n',
                '-sS',
                '-sU',
                '-T2',
                '-p',
                ('T:23,80,443,554,1883,7547,8080,8443,8883,37777,49152,U:69,161,1900,3671,5353,5683'),
                '-sV',
                '-O',
                '--osscan-limit',
                '--script',
                (
                    'upnp-info,'
                    'broadcast-upnp-info,'
                    'dns-service-discovery,'
                    'broadcast-dns-service-discovery,'
                    'coap-resources,'
                    'snmp-info,'
                    'snmp-sysdescr,'
                    'snmp-interfaces,'
                    'snmp-netstat,'
                    'tftp-enum,'
                    'tftp-version,'
                    'bacnet-info,'
                    'knx-gateway-info,'
                    'knx-gateway-discover,'
                    'modbus-discover,'
                    'ssl-cert,'
                    'ssl-enum-ciphers,'
                    'http-auth-finder,'
                    'http-title,'
                    'http-headers,'
                    'http-enum,'
                    'banner'
                ),
            ),
        ),
    ),
)


class PortSummary(BaseModel):
    protocol: str
    port: int
    state: str
    service: str | None = None
    product: str | None = None
    version: str | None = None
    extrainfo: str | None = None


class HostSummary(BaseModel):
    address: str
    hostnames: list[str] = Field(default_factory=list)
    state: str | None = None
    os_matches: list[str] = Field(default_factory=list)
    ports: list[PortSummary] = Field(default_factory=list)


class ScanSummary(BaseModel):
    profile: str
    stage: int
    stage_name: str
    target: str
    started_at: str
    finished_at: str
    command: list[str]
    hosts: list[HostSummary] = Field(default_factory=list)


def setup_logger(log_path: Path, verbose: bool = True) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger('scanner')

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ensure_nmap_exists(nmap_bin: str) -> None:
    if shutil.which(nmap_bin) is None:
        raise FileNotFoundError(f"Could not find Nmap binary '{nmap_bin}' in PATH.")


def sanitize_target_for_filename(target: str) -> str:
    return target.replace('/', '_').replace(':', '_')


def build_script_selector(
    scripts: list[str],
    aggressive: bool = False,
) -> str:
    """
    Build an --script expression.

    Conservative: uses explicit script list from stage definition.
    Aggressive: adds 'default' category + intrusive-only scripts.
    """
    parts: list[str] = list(scripts)

    if aggressive:
        # Add the 'default' category (works on Nmap 7.95)
        if 'default' not in parts:
            parts.insert(0, 'default')

        # Add intrusive-only scripts
        aggressive_intrusive = [
            'mqtt-subscribe',
            'rtsp-url-brute',
            'snmp-brute',
            'http-default-accounts',
        ]
        parts += [s for s in aggressive_intrusive if s not in parts]

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped = [p for p in parts if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

    return ','.join(deduped)


def build_common_args(config: CliConfig) -> list[str]:
    common: list[str] = []

    if config.skip_ping:
        common.append('-Pn')

    if config.only_open:
        common.append('--open')

    common.extend([
        '--reason',
        '--stats-every',
        config.stats_every,
        '--host-timeout',
        config.host_timeout,
    ])

    return common


def expand_targets(
    raw_targets: list[str],
    max_targets: int,
    exclude: set[str] | None = None,
) -> list[str]:
    expanded: list[str] = []
    exclude = exclude or set()

    for raw in raw_targets:
        if '/' in raw:
            network = ipaddress.ip_network(raw, strict=False)
            for host in network.hosts():
                host_str = str(host)
                if host_str not in exclude:
                    expanded.append(host_str)
        else:
            if raw not in exclude:
                expanded.append(raw)

    if len(expanded) > max_targets:
        raise ValueError(
            f'Expanded target count {len(expanded)} exceeds limit '
            f'({max_targets}). Reduce the CIDR scope or raise the limit.'
        )

    return expanded


def get_stage(profile: ScanProfile, stage_id: int) -> ScanStage:
    for stage in profile.stages:
        if stage.stage_id == stage_id:
            return stage
    raise KeyError(f'Unknown stage: {stage_id}')


def build_nmap_command(
    *,
    config: CliConfig,
    stage: ScanStage,
    target: str,
    output_base: Path,
) -> list[str]:
    cmd: list[str] = []

    if config.sudo:
        cmd.append('sudo')

    cmd.append(config.nmap_bin)
    cmd.extend(build_common_args(config))

    stage_args_list = list(stage.args)

    if config.aggressive and stage.stage_id == 3:
        try:
            script_idx = stage_args_list.index('--script')
            del stage_args_list[script_idx : script_idx + 2]
        except (ValueError, IndexError):
            pass

        script_selector = build_script_selector(
            scripts=[],
            aggressive=True,
        )
        stage_args_list.extend(['--script', script_selector])

        merged_args: dict[str, str] = {}
        merged_args.update(AGGRESSIVE_SCRIPT_ARGS)
        merged_args.update(config.script_args)

        if merged_args:
            args_str = ','.join(f'{k}={v}' for k, v in merged_args.items())
            stage_args_list.extend(['--script-args', args_str])

    cmd.extend(stage_args_list)
    cmd.extend(['-oA', str(output_base), target])

    return cmd


def run_command(
    cmd: list[str],
    dry_run: bool,
    logger: logging.Logger,
) -> subprocess.CompletedProcess[str] | None:
    logger.debug('Executing command: %s', ' '.join(cmd))

    if dry_run:
        logger.info('[DRY RUN] %s', ' '.join(cmd))
        return None

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
    )

    logger.debug('Command exit code: %s', result.returncode)

    if result.stdout.strip():
        logger.debug('Command stdout:\n%s', result.stdout.strip())

    if result.stderr.strip():
        logger.debug('Command stderr:\n%s', result.stderr.strip())

    return result


def parse_nmap_xml(xml_path: Path, logger: logging.Logger) -> list[HostSummary]:
    if not xml_path.exists():
        logger.warning('XML output file not found: %s', xml_path)
        return []

    logger.debug('Parsing XML file: %s', xml_path)
    logger.debug('XML file size: %d bytes', xml_path.stat().st_size)

    try:
        tree = et.parse(xml_path)
    except et.ParseError as exc:
        logger.exception('Failed to parse XML file %s: %s', xml_path, exc)
        return []

    root = tree.getroot()
    hosts: list[HostSummary] = []

    host_elems = root.findall('host')
    logger.debug('Found %d <host> entries in XML', len(host_elems))

    for host_elem in host_elems:
        address = ''
        addr_elem = host_elem.find('address')
        if addr_elem is not None:
            address = addr_elem.attrib.get('addr', '')

        state = None
        status_elem = host_elem.find('status')
        if status_elem is not None:
            state = status_elem.attrib.get('state')

        hostnames: list[str] = []
        hostnames_elem = host_elem.find('hostnames')
        if hostnames_elem is not None:
            for h in hostnames_elem.findall('hostname'):
                name = h.attrib.get('name')
                if name:
                    hostnames.append(name)

        os_matches: list[str] = []
        os_elem = host_elem.find('os')
        if os_elem is not None:
            for match in os_elem.findall('osmatch'):
                name = match.attrib.get('name')
                if name:
                    os_matches.append(name)

        ports: list[PortSummary] = []
        ports_elem = host_elem.find('ports')
        if ports_elem is not None:
            for port_elem in ports_elem.findall('port'):
                protocol = port_elem.attrib.get('protocol', '')
                portid = int(port_elem.attrib.get('portid', '0'))

                state_elem = port_elem.find('state')
                port_state = state_elem.attrib.get('state', '') if state_elem is not None else ''

                service_elem = port_elem.find('service')
                service = service_elem.attrib.get('name') if service_elem is not None else None
                product = service_elem.attrib.get('product') if service_elem is not None else None
                version = service_elem.attrib.get('version') if service_elem is not None else None
                extrainfo = service_elem.attrib.get('extrainfo') if service_elem is not None else None

                ports.append(
                    PortSummary(
                        protocol=protocol,
                        port=portid,
                        state=port_state,
                        service=service,
                        product=product,
                        version=version,
                        extrainfo=extrainfo,
                    )
                )

        logger.debug(
            'Parsed host address=%s state=%s hostnames=%s ports=%d',
            address,
            state,
            hostnames,
            len(ports),
        )

        hosts.append(
            HostSummary(
                address=address,
                hostnames=hostnames,
                state=state,
                os_matches=os_matches,
                ports=ports,
            )
        )

    logger.info('Parsed %d host(s) from XML %s', len(hosts), xml_path.name)
    return hosts


def write_json_summary(
    summary: ScanSummary,
    json_path: Path,
    logger: logging.Logger,
) -> None:
    json_path.write_text(
        json.dumps(summary.model_dump(mode='json'), indent=2),
        encoding='utf-8',
    )
    logger.debug('Wrote JSON summary: %s', json_path)


def log_scan_result(
    *,
    target: str,
    stage: ScanStage,
    result: subprocess.CompletedProcess[str] | None,
    summary: ScanSummary,
    logger: logging.Logger,
) -> None:
    logger.info('[%s] Stage %d: %s', target, stage.stage_id, stage.name)
    logger.info('Description: %s', stage.description)
    logger.info('Hosts parsed: %d', len(summary.hosts))

    open_ports = sum(1 for host in summary.hosts for port in host.ports if port.state == 'open')
    logger.info('Open ports parsed: %d', open_ports)

    if summary.hosts:
        for host in summary.hosts:
            logger.debug(
                'Host summary: address=%s state=%s hostnames=%s os_matches=%s',
                host.address,
                host.state,
                host.hostnames,
                host.os_matches,
            )
            for port in host.ports:
                logger.debug(
                    'Port: %s/%d state=%s service=%s product=%s version=%s extrainfo=%s',
                    port.protocol,
                    port.port,
                    port.state,
                    port.service,
                    port.product,
                    port.version,
                    port.extrainfo,
                )

    if result is not None:
        logger.info('Exit code: %d', result.returncode)
        if result.stderr.strip():
            logger.warning('stderr:\n%s', result.stderr.strip())


def run_scans(config: CliConfig, profile: ScanProfile) -> int:
    ensure_nmap_exists(config.nmap_bin)

    targets = expand_targets(config.targets, config.max_targets_per_run, set(config.exclude))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    run_dir = config.output_dir / f'run_{run_timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / 'scanner.log'
    logger = setup_logger(log_path)

    logger.info('Output directory: %s', run_dir)
    logger.info('Targets: %s', ', '.join(targets))
    logger.info('Stages: %s', ', '.join(str(s) for s in config.stages))
    logger.info('Profile: %s', profile.name)
    logger.info('Log file: %s', log_path)

    overall_exit_code = 0

    for target in targets:
        target_dir = run_dir / sanitize_target_for_filename(target)
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info('Processing target: %s', target)

        for stage_id in config.stages:
            stage = get_stage(profile, stage_id)
            output_base = target_dir / f'stage{stage.stage_id}_{stage.name}'
            cmd = build_nmap_command(
                config=config,
                stage=stage,
                target=target,
                output_base=output_base,
            )

            started_at = now_iso()
            result = run_command(cmd, config.dry_run, logger)
            finished_at = now_iso()

            xml_path = output_base.with_suffix('.xml')
            json_path = output_base.with_suffix('.summary.json')

            hosts = [] if config.dry_run else parse_nmap_xml(xml_path, logger)

            summary = ScanSummary(
                profile=profile.name,
                stage=stage.stage_id,
                stage_name=stage.name,
                target=target,
                started_at=started_at,
                finished_at=finished_at,
                command=cmd,
                hosts=hosts,
            )

            write_json_summary(summary, json_path, logger)
            log_scan_result(
                target=target,
                stage=stage,
                result=result,
                summary=summary,
                logger=logger,
            )

            if result is not None and result.returncode != 0:
                overall_exit_code = result.returncode

    logger.info('Completed all scans with overall exit code: %d', overall_exit_code)
    return overall_exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=('Safe staged Nmap automation for authorized IoT research environments.')
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        required=True,
        help='One or more IPs or CIDRs to scan.',
    )
    parser.add_argument(
        '--output-dir',
        default='nmap_scans',
        help='Directory where scan artifacts will be written.',
    )
    parser.add_argument(
        '--stages',
        nargs='+',
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help='Stages to run. Example: --stages 1 2',
    )
    parser.add_argument(
        '--nmap-bin',
        default='nmap',
        help='Path or name of the Nmap binary.',
    )
    parser.add_argument(
        '--sudo',
        action='store_true',
        help='Prefix the Nmap command with sudo.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without executing them.',
    )
    parser.add_argument(
        '--stats-every',
        default='30s',
        help='Nmap progress reporting interval.',
    )
    parser.add_argument(
        '--host-timeout',
        default='10m',
        help='Maximum time to spend per host.',
    )
    parser.add_argument(
        '--include-closed',
        action='store_true',
        help='Do not use --open, so closed/filtered ports stay in output too.',
    )
    parser.add_argument(
        '--no-skip-ping',
        action='store_true',
        help='Do not use -Pn. By default, this script uses -Pn for lab reliability.',
    )
    parser.add_argument(
        '--max-targets-per-run',
        type=int,
        default=256,
        help='Safety cap after CIDR expansion.',
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help='One or more IPs to exclude from scanning.',
    )
    parser.add_argument(
        '--aggressive',
        action='store_true',
        help=(
            'Enable intrusive scanning: adds brute, exploit, dos, intrusive '
            'NSE categories plus mqtt-subscribe (#), rtsp-url-brute, '
            'snmp-brute, http-default-accounts, and safe discovery scripts.'
        ),
    )
    parser.add_argument(
        '--script-args',
        nargs='+',
        default=[],
        help=('Override script args as KEY=VALUE pairs. Example: --script-args mqtt-subscribe.listen-time=60s'),
    )

    return parser


def _parse_script_args(arg_list: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE pairs from command line into a dict."""
    result: dict[str, str] = {}
    for item in arg_list:
        if '=' in item:
            key, val = item.split('=', 1)
            result[key] = val
    return result


def parse_config(argv: list[str]) -> CliConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    raw = {
        'targets': args.targets,
        'output_dir': Path(args.output_dir),
        'stages': args.stages,
        'nmap_bin': args.nmap_bin,
        'sudo': args.sudo,
        'dry_run': args.dry_run,
        'stats_every': args.stats_every,
        'host_timeout': args.host_timeout,
        'only_open': not args.include_closed,
        'skip_ping': not args.no_skip_ping,
        'max_targets_per_run': args.max_targets_per_run,
        'exclude': args.exclude,
        'aggressive': args.aggressive,
        'script_args': _parse_script_args(args.script_args),
    }

    try:
        return CliConfig.model_validate(raw)
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc


def main() -> int:
    config = parse_config(sys.argv[1:])
    return run_scans(config, SAFE_IOT_PROFILE)


if __name__ == '__main__':
    raise SystemExit(main())
