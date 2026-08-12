# Raspberry Pi 5 + NCNN สำหรับแอปถ่าย dataset

## สรุปฉบับสั้น
สำหรับ Raspberry Pi 5 RAM 8GB / SD 64GB การใช้งานที่ปลอดภัยและเร็วที่สุดคือ:

- ใช้ YOLOv8n หรือโมเดลเล็ก ๆ
- Export model จากเครื่อง PC / Colab เป็น NCNN
- คัดลอกไฟล์ model ไปที่ Pi
- รัน inference บน Pi แบบ real-time
- ใช้ OpenCV + camera preview + square crop 1:1 + save 224x244

สิ่งนี้เหมาะกว่าสำหรับบอร์ดจริง เพราะไม่ต้องใช้ full PyTorch runtime บน Pi

## เหตุผลที่เลือก NCNN
- เหมาะกับ ARM CPU บน Raspberry Pi
- ใช้ RAM และ SD น้อยกว่า PyTorch มาก
- นำไป deploy บนอุปกรณ์จริงได้ง่ายกว่า
- เหมาะกับการ deploy model แบบ lightweight

## แนะนำโมเดล
สำหรับงานถ่าย dataset เม็ดยา ให้ใช้:
- YOLOv8n (แนะนำมากที่สุด)
- หรือ MobileNet-SSD / custom tiny detector ถ้าต้องการเร็วขึ้นอีก

## Workflow ที่แนะนำ

### 1) Export โมเดลบนเครื่องที่มี GPU หรือ PC
```
python3 -m venv venv
source venv/bin/activate
pip install ultralytics

# ตัวอย่าง export เป็น ncnn
yolo export model=yolov8n.pt format=ncnn imgsz=640
```

ผลลัพธ์ที่ได้จะคล้าย:
- `yolov8n.param`
- `yolov8n.bin`

### 2) คัดลอก model ไปยัง Raspberry Pi
```
scp -r yolov8n_ncnn_model pi@<raspberrypi-ip>:/home/pi/Documents/Test/models/
```

### 3) ติดตั้ง dependencies บน Pi
```
sudo apt update
sudo apt install -y python3-opencv python3-pip
python3 -m pip install numpy
```

### 4) ถ้าใช้ Python wrapper ของ NCNN
ตรวจสอบว่ามี package ให้ใช้ หรือ install ถ้ามี
```
python3 -m pip install ncnn
```

ถ้า package นี้ไม่มี ให้ใช้โหมด C++ runtime หรือ binary อย่างเดียว

## โครงสร้างแอปที่ควรมี

```text
Test/
  models/
    yolov8n.param
    yolov8n.bin
  ncnn_dataset_capture.py
  requirements.txt
```

## ฟังก์ชันหลักของแอป
1. เปิดกล้อง real-time
2. ทดสอบ YOLO NCNN model
3. วาด bounding box บนวัตถุ
4. ปรับกรอบเป็นสี่เหลี่ยม 1:1 ให้ครอบวัตถุทั้งอัน
5. ตรวจสอบว่า object ทั้งอันอยู่ในกรอบก่อน capture
6. ถ่ายภาพแล้ว save เป็น 224x244
7. ตั้งชื่อไฟล์ prefix + running number ต่อเนื่อง
8. บันทึกลงโฟลเดอร์ที่เลือก

## หลักการ crop ที่ถูกต้อง
การทำงานที่ถูกต้องคือ:

- ใช้ YOLO detect วัตถุ
- ดึง bounding box
- ปรับ bounding box ให้เป็น square 1:1
- ขยับกรอบให้ครอบคลุมวัตถุทั้งอัน
- ตรวจว่า object ยังอยู่ภายในกรอบ
- จากนั้นจึง crop และ save เป็น 224x244

> ไม่ใช่การบีบภาพแบบเอาเอาๆ เพื่อให้ตรง 224x244 แบบดิบ ๆ

## ตัวเลือกที่ดีที่สุดสำหรับบอร์ดจริง
### เลือก 1: NCNN + Python
- ดีถ้ามี Python wrapper ที่ทำงานได้
- เหมาะกับ prototype และโพรเจกต์เร็ว

### เลือก 2: C++ NCNN runtime
- เร็วที่สุด
- เหมาะถ้าต้องใช้งานแบบ production
- เหมาะสำหรับงาน dataset ที่ต้อง speed สูง

## คำแนะนำของผมสำหรับคุณ
สำหรับ Raspberry Pi 5 ของคุณ ผมแนะนำว่า:

- ใช้YOLOv8n
- Export เป็น NCNN บน PC
- Deploy บน Pi
- ใช้ Python + OpenCV สำหรับกล้องและ crop
- ใช้ NCNN สำหรับ inference เท่านั้น

นี่เป็นแนวทางที่สอดคล้องกับบอร์ดจริงที่สุด และไม่ “ทำเกินไป” ให้ใช้พลังของ Pi อย่างเสียเปล่า

## สรุป
ถ้าคุณเลือกทาง NCNN แล้ว ผมจะทำต่อเป็น 3 ส่วนต่อไป:
1. สร้าง script export model สำหรับ PC
2. สร้าง script capture + crop สำหรับ Pi
3. สร้าง pipeline save dataset 224x244 แบบ sequence และเลือก folder ได้
