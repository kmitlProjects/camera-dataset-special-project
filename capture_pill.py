from picamera2 import Picamera2
import time

# เริ่มต้นการทำงานของกล้อง
picam2 = Picamera2()
picam2.start()

# รอให้กล้องปรับแสงและโฟกัสสักครู่
print("กำลังปรับแสง...")
time.sleep(2) 

# ถ่ายภาพและบันทึก
filename = "current_pill.jpg"
picam2.capture_file(filename)
print(f"ถ่ายภาพสำเร็จ! บันทึกไว้ที่: {filename}")

# ปิดกล้องเมื่อเสร็จสิ้น
picam2.stop()
