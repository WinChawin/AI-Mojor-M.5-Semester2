import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  

df = pd.read_csv("digit-recognizer/train.csv")
pix = [c for c in df.columns if c != 'label']                                                                 # ชื่อคอลัมน์พิกเซล

df = df.replace(r'^\s*$', np.nan, regex=True)                                                                 # แทนที่ช่องว่างให้เป็น NaN 
df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(pd.to_numeric, errors='coerce')     # อะไรที่ไม่ใช่ให้เป็น NaN
miss = df.index[df.isna().any(axis=1)]                                                                        # เก็บแถวที่มี NaN อย่างน้อย 1 column

p = df[pix].select_dtypes(include=[np.number])                                                                # เลือกเฉพาะ column pixelที่เป็นตัวเลข
oor = ((p < 0) | (p > 255)).any(axis=1).reindex(df.index, fill_value=False)                                   # หาว่ามีแถวไหนมีค่าพิกเซลนอกช่วง 0–255 แล้วทำให้ index ตรงกับ df เดิม
bad = ~df['label'].isin(range(10))                                                                            # เช็กว่า label ไม่ใช่ตัวเลข 0–9 รึป่าว
rng = df.index[oor | bad]                                                                                     # รวม index ของแถวที่พิกเซลหรือ label ผิดช่วง

h = pd.util.hash_pandas_object(df[pix].fillna(-1), index=False)                                               # สร้างแฮชของพิกเซลหาภาพซ้ำ (เติม NaN ด้วย -1)
df['_h'] = h                                                                                                  # เพิ่ม column ใหม่เก็บรหัสแฮชไว้ใช้ตรวจซ้ำ
g = df.groupby('_h')                                                                                          # จัดกลุ่มแถวที่มีรหัสแฮชเหมือนกัน-ภาพซ้ำกัน
conf = g['label'].nunique(dropna=True) > 1                                                                    # ดูว่ากลุ่มไหนมี label หลายค่า (ภาพเดียวกันแต่ติดคนละเลข)
drop_conf = df.index[df['_h'].isin(conf[conf].index)]                                                         # เอา index ของภาพที่ label ขัดแย้งไว้ลบทิ้งทั้งหมด
drop_dup = df.index[df.duplicated('_h') & ~df['_h'].isin(conf[conf].index)]                                   # เอา index ของภาพที่ซ้ำกันแต่ label เดียวกัน

bad_idx = set(miss) | set(rng) | set(drop_conf) | set(drop_dup)                                               # รวมที่มีปัญหาทุกแบบ
clean = df.drop(index=list(bad_idx)).drop(columns=['_h'])                                                     # ลบ row ปัญหา และทิ้งคอลัมน์แฮชชั่วคราว _h
clean.to_csv("digit-recognizer/train_clean.csv", index=False)                                                 # เซฟไฟล์ใหม่

print(f"original: {len(df)} | cleaned: {len(clean)} | dropped: {len(df)-len(clean)}")                         # สรุปจำนวน rowก่อน-หลัง, จำนวนที่ถูกลบ

y_train = clean["label"]
x_train = clean.drop('label', axis=1)

plt.figure(figsize=(6, 12))                                                                                   # (เพิ่มขนาดภาพให้พอดีกริด)
for x in range(10):
    if x in y_train.values:                                                                                   # มี label นี้ไหม
        plt.subplot(5, 2, x + 1)                                                                              # กริด 5x2
        plt.imshow(x_train[y_train == x].values[0].reshape(28, 28), cmap='gray')
        plt.title(str(x))
        plt.axis('off')
plt.tight_layout()
plt.show()
