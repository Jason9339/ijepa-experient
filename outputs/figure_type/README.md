---

# CoSyn / Diagram：figure\_type 統計與圖片配對輸出

本資料夾由 `export_figure_type_stats.py` 產生，用來彙整 **figure\_type 統計** 與 **圖片 × 類別配對**。

* **來源資料集**：`/ceph/work/KLP/zihcilin39/datasets/CoSyn/diagram`
* **本資料夾位置**：`/ceph/work/KLP/zihcilin39/ijepa-experient/outputs/figure_type`
* 產出記錄可見於 Slurm log：`ijepa-experient/slurm-logs/<job_id>.out`

---

## 檔案一覽

* `figure_type_counts.tsv`
  每個 `figure_type` 的計數（TSV，tab 分隔）。
* `figure_type_counts_sorted.tsv`
  由大到小排序之計數（TSV）。
* `figure_type_distribution.png`
  `figure_type` 分佈長條圖（若執行環境未裝 `matplotlib` 可能不產生）。
* `figure_type_index.json`
  摘要與完整索引：

  * `summary`: 樣本總數、類別數
  * `categories`: `[{id, name, count}]`
  * `name_to_id`: 類別名稱 → 類別 id
  * `ids_by_category`: 類別 id（字串） → 該類別的 `id` 清單（可受 `--max_ids_per_type` 限制）
* `pairs.tsv`
  每筆樣本一列（tab 分隔）：
  `id  figure_type  image_path  width  height`

  > 多數 HF 影像資料集不含原始檔路徑，`image_path` 可能為空。若需要檔案，可用 `--export_images` 另存 PNG 到 `images/` 目錄（執行時指定才會生成）。
* `images/`（可選）
  僅在執行時帶 `--export_images` 時產生，存放回填之 PNG。

---

## 檔案格式

### 1) `figure_type_counts*.tsv`

* **分隔符號**：Tab（`\t`）
* **欄位**：

  * `figure_type`（字串；已做去頭尾與多重空白壓縮）
  * `count`（整數）

**前 10 名查看範例：**

```bash
tail -n +2 figure_type_counts_sorted.tsv | head -10 | column -s $'\t' -t
```

### 2) `figure_type_index.json`

```json
{
  "summary": {
    "total_samples": 34963,
    "unique_figure_types": 42
  },
  "categories": [
    {"id": 0, "name": "Textbook Diagram", "count": 4050},
    {"id": 1, "name": "Machine Learning Diagram", "count": 1565}
  ],
  "name_to_id": {
    "Textbook Diagram": 0,
    "Machine Learning Diagram": 1
  },
  "ids_by_category": {
    "0": ["<sample_id_1>", "<sample_id_2>"],
    "1": ["<sample_id_3>"]
  }
}
```

> 若執行時有加 `--max_ids_per_type N`，`ids_by_category` 每類只會保留前 `N` 筆以控制檔案大小。

### 3) `pairs.tsv`

* **分隔符號**：Tab（`\t`）
* **欄位**：`id`、`figure_type`、`image_path`、`width`、`height`
* `image_path` 可能為空；帶 `--export_images` 會另存為 `images/<id>.png` 並回填。

**檢視範例：**

```bash
# 看前 5 列（tab 對齊）
head -5 pairs.tsv | column -s $'\t' -t

# 查某類別的數筆樣本（例如 Flow Chart）
awk -F'\t' '$2=="Flow Chart"{print $0}' pairs.tsv | head -5
```

---

## 快速檢查

**1) counts 加總是否等於總樣本數**

```bash
awk -F'\t' 'NR>1{s+=$2} END{print s}' figure_type_counts.tsv
```

**2) pairs 行數應為（樣本數 + 1 表頭）**

```bash
wc -l pairs.tsv
```

**3) 用 pairs 聚合後與 counts 對比（無 diff 表示一致）**

```bash
tail -n +2 pairs.tsv | cut -f2 | sort | uniq -c \
| awk '{print substr($0, index($0,$2))"\t"$1}' | sort -t $'\t' -k1,1 > /tmp/from_pairs.tsv
tail -n +2 figure_type_counts.tsv | sort -t $'\t' -k1,1 > /tmp/from_counts.tsv
diff -u /tmp/from_counts.tsv /tmp/from_pairs.tsv | head -50
```

**4) 查看前 10 類別（注意 TSV 是 tab 分隔）**

```bash
tail -n +2 figure_type_counts_sorted.tsv | head -10 | column -s $'\t' -t
```

---

## 常用查詢

**某類別抽幾筆看看（以 Flow Chart 為例）**

```bash
awk -F'\t' '$2=="Flow Chart"{print $0}' pairs.tsv | head -5 | column -s $'\t' -t
```

**統計缺 `image_path` 的列數（資訊性檢查）**

```bash
awk -F'\t' 'NR>1{if($3=="") m++} END{print "missing image_path:", m}' pairs.tsv
```

**統計缺尺寸的列數（理應極少）**

```bash
awk -F'\t' 'NR>1{if($4=="" || $5=="") z++} END{print "rows missing size:", z}' pairs.tsv
```

---

## 參數總表（`export_figure_type_stats.py`）

* `--out_dir PATH`：輸出目錄（本資料夾）。
* `--no_plot`：不畫圖（即使環境有 `matplotlib`）。
* `--max_ids_per_type N`：`figure_type_index.json` 中每類最多保留 N 個 `id`。
* `--no_pairs`：跳過 `pairs.tsv`。
* `--pairs_limit_per_type N`：`pairs.tsv` 每類最多輸出 N 筆。
* `--export_images`：當 `image_path` 缺失時，另存 PNG 至 `images/`，並在 `pairs.tsv` 回填路徑。

---

## 備註

* 所有 `.tsv` 皆為 **tab 分隔**；排序與排版請使用 `-t $'\t'` 與 `column -s $'\t' -t`。
* 類別名稱已做基本正規化（去頭尾與多重空白壓縮），例如 `"Bottom    Up  Flow   Chart"` → `"Bottom Up Flow Chart"`。
* 第一次在新節點畫圖若顯示 font-cache 訊息屬正常。若想加速後續執行，可在 job 檔加入：

  ```bash
  export MPLBACKEND=Agg
  export MPLCONFIGDIR="${HOME}/.cache/matplotlib"
  mkdir -p "$MPLCONFIGDIR"
  ```

