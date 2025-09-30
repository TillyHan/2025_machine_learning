# make_datasets.py
import argparse
import xml.etree.ElementTree as ET
import csv
import re
from pathlib import Path

# 作業給定的座標與網格規格
START_LON = 120.00
START_LAT = 21.88
STEP = 0.03
NX = 67   # 每列經向點數（先經度遞增）
NY_EXPECT = 120  # 應有的列數（緯向）

NS = {"cwa": "urn:cwa:gov:tw:cwacommon:0.1"}  # CWA namespace

def parse_values(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    # 先按 namespace 正規找
    elem = root.find(".//cwa:dataset/cwa:Resource/cwa:Content", NS)
    # 保險：若上行失敗，抓任何結尾為 Content 的標籤
    if elem is None:
        elem = next((e for e in root.iter() if e.tag.endswith("Content")), None)
    if elem is None or not (elem.text or "").strip():
        raise RuntimeError("找不到 <Content> 或內容為空")

    # 逗號或空白分割；可解析 28.1E+00、40.0E-03 之類科學記號
    tokens = [t for t in re.split(r"[,\s]+", elem.text.strip()) if t]
    values = [float(t) for t in tokens]
    return values

def to_lon_lat(i, j):
    """i: 0..NX-1 (eastward), j: 0..NY-1 (northward)"""
    return START_LON + i * STEP, START_LAT + j * STEP

def main(xml_file: str, outdir: str):
    xml_path = Path(xml_file)
    out_dir = Path(outdir); out_dir.mkdir(parents=True, exist_ok=True)

    values = parse_values(xml_path)
    total = len(values)
    if total % NX != 0:
        raise ValueError(f"資料長度 {total} 不是 {NX} 的倍數，無法切成每列 {NX} 筆。")
    ny = total // NX
    if ny != NY_EXPECT:
        print(f"[警告] 解析到的列數 = {ny}，與預期 {NY_EXPECT} 不同（以解析值為準）。")

    # (a) Classification
    cls_path = out_dir / "classification.csv"
    with cls_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "label"])
        for j in range(ny):
            for i in range(NX):
                v = values[j * NX + i]
                lon, lat = to_lon_lat(i, j)
                label = 0 if v == -999.0 else 1
                w.writerow([f"{lon:.4f}", f"{lat:.4f}", label])

    # (b) Regression（只保留有效值）
    reg_path = out_dir / "regression.csv"
    with reg_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "value"])
        for j in range(ny):
            for i in range(NX):
                v = values[j * NX + i]
                if v == -999.0:
                    continue
                lon, lat = to_lon_lat(i, j)
                w.writerow([f"{lon:.4f}", f"{lat:.4f}", f"{v:.2f}"])

    valid_cnt = sum(v != -999.0 for v in values)
    print("完成 ✅")
    print(f"- 讀入格點數：{total} ({NX} × {ny})")
    print(f"- 有效值筆數（回歸集）：{valid_cnt}")
    print(f"- 輸出：{cls_path}")
    print(f"- 輸出：{reg_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="O-A0038-003 轉成 classification/regression 資料集")
    ap.add_argument("--xml", default="O-A0038-003.xml", help="XML 檔名或路徑")
    ap.add_argument("--out", default=".", help="輸出資料夾")
    args = ap.parse_args()
    main(args.xml, args.out)
