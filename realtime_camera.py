import cv2
from picamera2 import Picamera2

# 1. เริ่มต้นการทำงานของกล้อง
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

print("เปิดกล้องเรียลไทม์สำเร็จ!")
print("========================================")
print(" 📸 นำเมาส์ไปคลิกที่หน้าต่างกล้อง 1 ครั้งก่อนกดปุ่ม")
print(" [Spacebar] หรือ [c] : เพื่อถ่ายภาพและบันทึก")
print(" [q]                 : เพื่อออกและปิดกล้อง")
print("========================================")

img_counter = 1 # ตัวนับจำนวนรูปภาพ เพื่อไม่ให้ชื่อไฟล์ซ้ำกันตอนเซฟ

try:
    while True:
        # 2. ดึงภาพและแปลงสี
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 3. แสดงหน้าต่างภาพ
        cv2.imshow("Pill Detection - Realtime", frame_bgr)

        # 4. ดักจับการกดปุ่มบนคีย์บอร์ด (รอ 1 มิลลิวินาที)
        key = cv2.waitKey(1) & 0xFF

        # ถ้ากดปุ่ม 'q' ให้หยุดการทำงาน
        if key == ord('q'):
            print("กำลังปิดกล้อง...")
            break
            
        # ถ้ากดปุ่ม Spacebar (รหัส 32) หรือตัว 'c' ให้ทำการบันทึกภาพ
        elif key == 32 or key == ord('c'):
            filename = f"pill_dataset_{img_counter}.jpg"
            cv2.imwrite(filename, frame_bgr) # บันทึกภาพลงเครื่อง
            print(f"✅ แชะ! ถ่ายภาพสำเร็จ บันทึกไว้ที่: {filename}")
            img_counter += 1 # เพิ่มตัวเลขไปอีก 1 รูปถัดไปจะได้เป็นชื่อ 2, 3, 4...

except KeyboardInterrupt:
    print("บังคับปิดการทำงาน...")

finally:
    # 5. คืนทรัพยากรระบบเสมอ
    cv2.destroyAllWindows()
    picam2.stop()
