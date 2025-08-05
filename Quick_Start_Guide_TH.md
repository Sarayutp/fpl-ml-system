# 🚀 คู่มือเริ่มต้นใช้งานเร็ว - FPL ML System

## ⚡ เริ่มใช้งานใน 5 นาที

### ขั้นตอนที่ 1: ติดตั้ง (2 นาที)
```bash
# Clone และติดตั้ง
git clone <repository-url>
cd fpl-ml-system
pip install -r requirements.txt
pip install -e .
```

### ขั้นตอนที่ 2: ตั้งค่า (2 นาที)
```bash
# คัดลอกและแก้ไขการตั้งค่า
cp .env.example .env

# แก้ไขใน .env:
FPL_TEAM_ID=123456              # รหัสทีม FPL ของคุณ
FPL_EMAIL=your@email.com        # อีเมล FPL  
OPENAI_API_KEY=sk-xxx...        # API Key (ถ้ามี)
```

### ขั้นตอนที่ 3: เริ่มใช้งาน (1 นาที)
```bash
# ทดสอบระบบ
fpl status

# ดูทีมปัจจุบัน
fpl team show

# หาคำแนะนำเทรด
fpl transfer suggest
```

---

## 🔥 คำสั่งที่ใช้บ่อยที่สุด

### 🏆 จัดการทีม
```bash
fpl team show              # ดูทีมปัจจุบัน
fpl team optimize          # หาทีมที่ดีที่สุด
fpl team history --weeks 5 # ดูประวัติ 5 สัปดาห์
```

### 🔄 วิเคราะห์เทรด
```bash
fpl transfer suggest                    # แนะนำเทรดทั่วไป
fpl transfer analyze "Salah" "KDB"     # เปรียบเทียบเทรดเฉพาะ
fpl transfer plan --weeks 4            # วางแผนเทรด 4 สัปดาห์
```

### 👤 ค้นหานักเตะ
```bash
fpl player search --position FWD --max-price 10  # หา Forward ราคาไม่เกิน 10M
fpl player compare "Haaland" "Kane"              # เปรียบเทียบนักเตะ
fpl player analyze "Salah" --weeks 3             # วิเคราะห์ Salah 3 สัปดาห์
```

### 🔮 AI Predictions
```bash
fpl prediction captain           # แนะนำกัปตัน
fpl prediction points "Kane"     # ทำนายคะแนน Kane
fpl prediction price --days 7    # ทำนายการเปลี่ยนราคา 7 วัน
```

---

## 🌐 เปิด Web Dashboard

```bash
# เริ่ม Dashboard
streamlit run src/dashboard/app.py

# เข้าใช้งานที่: http://localhost:8501
```

### 📱 หน้าจอใน Dashboard:
- **Overview** - สรุปทีมและสถิติ
- **Team Analysis** - วิเคราะห์ทีมละเอียด  
- **Transfers** - คำแนะนำเทรดจาก AI
- **Players** - ค้นหาและเปรียบเทียบนักเตะ
- **AI Insights** - การทำนายและแชทบอท
- **Performance** - วิเคราะห์ประสิทธิภาพ

---

## 💡 เคล็ดลับสำคัญ

### ✅ สิ่งที่ควรทำ
- อัปเดตข้อมูลทุกอังคาร: `fpl data update`
- ดูคำแนะนำก่อน Deadline: `fpl transfer suggest`  
- เลือกกัปตันด้วย AI: `fpl prediction captain`
- วางแผนระยะยาว: `fpl transfer plan --weeks 4`

### ❌ สิ่งที่ไม่ควรทำ
- ไม่ใช้ AI แทนการคิดเอง - ใช้เป็นเครื่องมือช่วย
- ไม่เทรดทุกสัปดาห์ - เก็บ Free Transfer ไว้ใช้
- ไม่ลืมติดตามข่าวอาการบาดเจ็บ
- ไม่ใช้ Wildcard โดยไม่มีแผน

---

## 🆘 แก้ไขปัญหาเร็ว

### ปัญหา: คำสั่งไม่ทำงาน
```bash
# ตรวจสอบการติดตั้ง
pip list | grep fpl-ml-system
fpl --version
```

### ปัญหา: ไม่สามารถดึงข้อมูล FPL
```bash
# ตรวจสอบการตั้งค่า
fpl data health
fpl configure check-credentials
```

### ปัญหา: AI ไม่ทำงาน
```bash
# ตรวจสอบ API Key
echo $OPENAI_API_KEY
fpl prediction test-connection
```

---

## 📋 ตัวอย่างการใช้งานรายสัปดาห์

### 🗓️ วันจันทร์ (หลังจบเกมส์)
```bash
fpl data update                    # อัปเดตข้อมูลล่าสุด
fpl team show                      # ดูผลประสิทธิภาพทีม  
fpl analysis performance           # วิเคราะห์ประสิทธิภาพ
```

### 🗓️ วันพุธ-พฤหัส (วางแผน)
```bash
fpl transfer suggest --weeks 3     # ดูคำแนะนำเทรด
fpl analysis fixtures --weeks 5    # วิเคราะห์ความยากของนัด
fpl player search --differential   # หานักเตะราคาถูกที่มีโอกาส
```

### 🗓️ วันศุกร์ (ก่อน Deadline)
```bash
fpl prediction captain --strategy balanced  # เลือกกัปตัน
fpl transfer analyze "PlayerOut" "PlayerIn" # ตัดสินใจเทรดสุดท้าย
fpl team optimize                           # เช็คทีมสุดท้าย
```

---

## 🎯 เป้าหมายและการติดตาม

### 📊 ตัวชี้วัดสำคัญ
```bash
fpl analysis rank              # ดูอันดับปัจจุบัน
fpl analysis trends            # ดูเทรนด์เปรียบเทียบ
fpl team history --summary     # สรุปประสิทธิภาพรวม
```

### 🏆 เป้าหมายแนะนำ
- **มือใหม่**: Top 1M ในครึ่งแรกของซีซั่น
- **มือกลาง**: Top 500K ตลอดซีซั่น  
- **มือโปร**: Top 100K และไล่เข้า Top 10K

---

## 🔗 ลิงก์มีประโยชน์

### 📚 เอกสารเพิ่มเติม
- [คู่มือการใช้งานฉบับเต็ม](คู่มือการใช้งาน.md)
- [คู่มือการติดตั้ง](CLAUDE.md)
- [ข้อมูลเทคนิค](INITIAL.md)

### 🌐 เว็บไซต์ที่เกี่ยวข้อง
- [Fantasy Premier League Official](https://fantasy.premierleague.com)
- [FPL Statistics](https://www.fplstatistics.co.uk)  
- [Fantasy Football Scout](https://www.fantasyfootballscout.co.uk)

---

## 🎉 พร้อมแล้ว!

ตอนนี้คุณมีเครื่องมือที่ทรงพลังในการจัดการทีม FPL แล้ว!

**ขั้นตอนถัดไป:**
1. รัน `fpl team show` เพื่อดูทีมปัจจุบัน
2. รัน `fpl transfer suggest` เพื่อดูคำแนะนำเทรด
3. เปิด Dashboard ด้วย `streamlit run src/dashboard/app.py`
4. มาดูกันว่าคุณจะไปถึง **Top 10K** ได้หรือไม่! 🚀

**Good luck และสนุกกับการเล่น FPL!** ⚽🏆