# 📂 DFAIR Dataset

This directory contains the **DFAIR dataset**, derived from controlled experiments on IoT devices under real-world attack scenarios. The dataset has been processed with **CICFlowMeter** to extract flow-level features for machine learning-based intrusion detection.

---

## 📃 Dataset File

* `combined_data_with_okpVacc_modified.csv`:

  * CSV-formatted network flow data
  * Generated from Wireshark PCAPs using CICFlowMeter
  * Includes flows from both benign and malicious sessions involving 10 IoT devices

---

## 🚀 Data Collection Methodology

IoT traffic was collected in a **controlled lab environment** using a TP-Link TL-WR541N router running OpenWrt. Each IoT device was connected to this router and configured using a Samsung Galaxy A71 smartphone. Devices were assigned static IPs for consistent packet capture.

Benign traffic was recorded during normal device usage. Each device underwent an 8-hour capture session, during which four attacks were launched:

* **TCP SYN Flood**
* **XMAS Tree Flood**
* **UDP Flood**
* **HTTP Flood**

These attacks were performed 3 times per device, producing a diverse set of malicious traffic samples.

---

## 📅 IoT Devices

| Device Name                   | Interaction Methods               |
| ----------------------------- | --------------------------------- |
| Amazon Echo Dot (5th Gen)     | Smartphone App, Voice Interaction |
| Kasa Smart Wi-Fi Plug Mini    | Smartphone App, Physical Button   |
| LongPlus X07 Baby Monitor     | Smartphone App                    |
| Ring Video Doorbell (2nd Gen) | Smartphone App, Device Controller |
| Google Nest Mini              | Smartphone App, Voice Interaction |
| Google Home Cam               | Smartphone App                    |
| NiteBird Smart Bulb           | Smartphone App                    |
| OKP K2 Robotic Vacuum         | Smartphone App                    |
| Roborock K2 Vacuum            | Smartphone App, Physical Button   |
| Philips Hue Hub               | Smartphone App                    |

---

## 🔢 Full Feature Set

Below is the complete list of flow features extracted using CICFlowMeter:

| Feature Name                                                                                                                     | Description                                             |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| device\_id                                                                                                                       | ID of the device generating the traffic                 |
| session\_id                                                                                                                      | ID of the session capture                               |
| src\_ip, dst\_ip                                                                                                                 | Source and destination IP addresses                     |
| src\_port, dst\_port                                                                                                             | Source and destination ports                            |
| protocol                                                                                                                         | Transport layer protocol                                |
| timestamp                                                                                                                        | Timestamp of the flow capture                           |
| flow\_duration                                                                                                                   | Duration of the flow in microseconds                    |
| flow\_byts\_s                                                                                                                    | Number of flow bytes per second                         |
| flow\_pkts\_s                                                                                                                    | Number of flow packets per second                       |
| fwd\_pkts\_s, bwd\_pkts\_s                                                                                                       | Forward and backward packet rates                       |
| tot\_fwd\_pkts, tot\_bwd\_pkts                                                                                                   | Total packets in forward/backward directions            |
| totlen\_fwd\_pkts, totlen\_bwd\_pkts                                                                                             | Total size of packets in forward/backward directions    |
| fwd\_pkt\_len\_max/min/mean/std                                                                                                  | Packet length stats (forward)                           |
| bwd\_pkt\_len\_max/min/mean/std                                                                                                  | Packet length stats (backward)                          |
| pkt\_len\_max/min/mean/std/var                                                                                                   | Packet length statistics (overall)                      |
| fwd\_header\_len, bwd\_header\_len                                                                                               | Header lengths in forward/backward directions           |
| fwd\_seg\_size\_min                                                                                                              | Minimum segment size observed (forward)                 |
| fwd\_act\_data\_pkts                                                                                                             | Active data packets in forward direction                |
| flow\_iat\_mean/max/min/std                                                                                                      | Inter-arrival time statistics (flow-level)              |
| fwd\_iat\_tot/max/min/mean/std                                                                                                   | Inter-arrival time stats (forward direction)            |
| bwd\_iat\_tot/max/min/mean/std                                                                                                   | Inter-arrival time stats (backward direction)           |
| fwd\_psh\_flags, bwd\_psh\_flags                                                                                                 | Number of PSH flags seen in forward/backward directions |
| fwd\_urg\_flags, bwd\_urg\_flags                                                                                                 | Number of URG flags seen in forward/backward directions |
| fin\_flag\_cnt, syn\_flag\_cnt, rst\_flag\_cnt, psh\_flag\_cnt, ack\_flag\_cnt, urg\_flag\_cnt, ece\_flag\_cnt, cwr\_flag\_count | TCP flag counts                                         |
| down\_up\_ratio                                                                                                                  | Ratio of downloaded to uploaded bytes                   |
| pkt\_size\_avg                                                                                                                   | Average size of packet                                  |
| init\_fwd\_win\_byts, init\_bwd\_win\_byts                                                                                       | Initial window sizes (forward/backward)                 |
| active\_max/min/mean/std                                                                                                         | Flow active duration stats                              |
| idle\_max/min/mean/std                                                                                                           | Flow idle duration stats                                |
| fwd\_byts\_b\_avg, fwd\_pkts\_b\_avg                                                                                             | Forward bulk average (bytes, packets)                   |
| bwd\_byts\_b\_avg, bwd\_pkts\_b\_avg                                                                                             | Backward bulk average (bytes, packets)                  |
| fwd\_blk\_rate\_avg, bwd\_blk\_rate\_avg                                                                                         | Bulk rate (forward/backward)                            |
| fwd\_seg\_size\_avg, bwd\_seg\_size\_avg                                                                                         | Segment size averages                                   |
| subflow\_fwd\_pkts/byts                                                                                                          | Subflow stats (forward)                                 |
| subflow\_bwd\_pkts/byts                                                                                                          | Subflow stats (backward)                                |
| Label                                                                                                                            | Class label (Benign or Malicious)                       |

---

## 🔒 Usage

This dataset is intended strictly for **research and educational purposes**. Redistribution or use in production environments is not permitted without explicit permission.

---

## 📢 Contact

Seth Barrett | [GitHub](https://github.com/sethbarrett50) | [sebarrett@augusta.edu](mailto:sebarrett@augusta.edu)
Bradley Boswell | [GitHub](https://github.com/bradleyboswell) | [brboswell@augusta.edu](mailto:brboswell@augusta.edu)
Swarnamugi Rajaganapathy, PhD | [GitHub](https://github.com/swarna6384) | [swarnamugi@dfairlab.com](mailto:swarnamugi@dfairlab.com)
Lin Li, PhD | [GitHub](https://github.com/linli786) | [lli1@augusta.edu](mailto:lli1@augusta.edu)
Gokila Dorai, PhD | [GitHub](https://github.com/gdorai) | [gdorai@augusta.edu](mailto:gdorai@augusta.edu)