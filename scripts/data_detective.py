"""
data_detective.py

A flexible pipeline for cleaning and merging Chinese provincial statistical
data (e.g. GDP, population) that is published in inconsistent formats
(varying encodings, delimiters, header structures, and footnote conventions).

This was originally written to replicate, in Python, a panel-data cleaning
and regression workflow first completed in Stata for an undergraduate
Econometrics course.

Pipeline stages:
    1. Scan a folder for raw CSV files
    2. Auto-detect file encoding and delimiter
    3. Identify the indicator each file represents (from footnotes or filename)
    4. Reshape each file from wide (year columns) to long (panel) format
    5. Merge all indicators into a single entity-year panel
    6. Clean the merged panel: remove duplicates, drop statistical outliers
       (3-sigma rule), validate domain constraints, interpolate missing values
    7. Run an OLS regression and export results + diagnostic plots

Output:
    data/processed/panel_data.csv
    output/cleaning_report.txt
    output/regression_results.txt
    output/panel_analysis.png
"""

import pandas as pd
import numpy as np
import os
import re
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("output")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DataDetective:
    """
    Cleans and merges raw, inconsistently formatted Chinese statistical
    CSVs into a single analysis-ready panel dataset.
    """

    def __init__(self, folder_path: str = str(RAW_DIR)):
        self.folder_path = folder_path
        self.files = []
        self.merged_data = None
        self.indicator_map = {}
        self.cleaning_report = {}

    # ------------------------------------------------------------------
    # 1. File discovery
    # ------------------------------------------------------------------
    def scan_files(self):
        self.files = [f for f in os.listdir(self.folder_path) if f.endswith(".csv")]
        print(f"发现 {len(self.files)} 个CSV文件")
        for f in self.files:
            print(f"  - {f}")
        return self.files

    # ------------------------------------------------------------------
    # 2. Encoding / delimiter detection
    # ------------------------------------------------------------------
    def detect_delimiter(self, filepath):
        for enc in ["utf-8", "gbk", "gb18030", "utf-8-sig"]:
            try:
                with open(filepath, "r", encoding=enc) as f:
                    f.readline()
                    f.readline()
                    third_line = f.readline().strip()

                delimiters = [",", "\t", "，", ";", "|"]
                best_delim, best_count = ",", 0
                for delim in delimiters:
                    count = third_line.count(delim)
                    if count > best_count:
                        best_count, best_delim = count, delim
                if best_count > 0:
                    return best_delim, enc
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        return ",", "utf-8"

    def read_and_parse_csv(self, filepath, header_row_index: int = 2):
        """
        Parse a raw statistics-bureau CSV.

        National Bureau of Statistics exports (data.stats.gov.cn) place two
        metadata lines before the real header — a database title line and a
        "时间：" line — so the header itself is on line index 2 (0-indexed).
        `header_row_index` lets the caller override this for bureau exports
        that don't include the metadata lines.
        """
        delim, enc = self.detect_delimiter(filepath)
        print(f"  探测到分隔符: '{delim}'")

        try:
            with open(filepath, "r", encoding=enc) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if not lines:
                return None, [], None

            data_lines, note_lines = [], []
            for i, line in enumerate(lines):
                if "注" in line:
                    note_lines.append(line)
                elif i >= header_row_index:
                    data_lines.append(line)

            if not data_lines:
                return None, note_lines, enc

            header = [h.strip() for h in data_lines[0].split(delim) if h.strip()]

            # Fall back to a better-fitting delimiter if the first guess
            # produced a degenerate single-column header.
            if len(header) <= 1:
                for alt_delim in [",", "\t", "，", ";"]:
                    if alt_delim != delim:
                        test_header = [
                            h.strip() for h in data_lines[0].split(alt_delim) if h.strip()
                        ]
                        if len(test_header) > len(header):
                            delim, header = alt_delim, test_header
                            print(f"  更换分隔符为: '{delim}'")
                            break

            rows = []
            for line in data_lines[1:]:
                parts = [p.strip() for p in line.split(delim)]
                if len(parts) < len(header):
                    parts += [""] * (len(header) - len(parts))
                rows.append(parts[: len(header)])

            df = pd.DataFrame(rows, columns=header)
            df.columns = [col.replace("\ufeff", "").strip() for col in df.columns]
            df = df.dropna(how="all", axis=1).dropna(how="all", axis=0).reset_index(drop=True)
            return df, note_lines, enc

        except Exception as e:
            print(f"  读取失败: {e}")
            return None, [], None

    # ------------------------------------------------------------------
    # 3. Indicator identification
    # ------------------------------------------------------------------
    def extract_indicator_from_note(self, note_lines, filename):
        if not note_lines:
            return self.extract_from_filename(filename)

        full_note = " ".join(note_lines)
        keywords = [
            ("GDP_亿元", ["生产总值", "GDP", "国内生产总值", "国民生产总值"]),
            ("人口_万人", ["常住人口", "户籍人口", "总人口", "人口数", "人口"]),
            ("教育经费_亿元", ["教育经费", "教育支出"]),
            ("研发投入_亿元", ["研发", "R&D", "科技经费"]),
            ("财政收入_亿元", ["财政收入", "一般公共预算"]),
            ("CPI_指数", ["CPI", "消费价格指数"]),
        ]
        for indicator, words in keywords:
            if any(word in full_note for word in words):
                return indicator
        return self.extract_from_filename(filename)

    def extract_from_filename(self, filename):
        name_map = {
            "国民生产总值": "GDP_亿元",
            "生产总值": "GDP_亿元",
            "总人口": "人口_万人",
            "人口": "人口_万人",
        }
        for key, value in name_map.items():
            if key in filename:
                return value
        return "未知指标"

    # ------------------------------------------------------------------
    # 4. Wide-to-long reshape (panel construction)
    # ------------------------------------------------------------------
    def wide_to_long(self, df, indicator_name):
        if df is None or len(df) == 0:
            return None

        region_col = df.columns[0]
        print(f"  地区列: '{region_col}'")

        year_cols = []
        for col in df.columns[1:]:
            col_str = str(col).strip()
            if "年" in col_str or re.match(r"^\d{4}$", col_str):
                year_cols.append(col)
            elif re.search(r"(\d{4})", col_str):
                year_cols.append(col)

        if not year_cols:
            print("  未找到年份列")
            return None

        print(f"  找到 {len(year_cols)} 个年份列")
        df_clean = df[[region_col] + year_cols].copy()
        for col in year_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        df_long = df_clean.melt(
            id_vars=[region_col], var_name="年份", value_name=indicator_name
        )
        # Standardize the region column name across all source files, since
        # different bureau exports label it differently (地区/省份/城市 etc.).
        # Downstream cleaning and regression code assumes a single "地区" column.
        df_long = df_long.rename(columns={region_col: "地区"})
        df_long["年份"] = df_long["年份"].astype(str).str.extract(r"(\d{4})")[0]
        df_long["年份"] = pd.to_numeric(df_long["年份"], errors="coerce")
        df_long = df_long.dropna(subset=["地区", "年份"])
        df_long = df_long[df_long[indicator_name].notna()]
        return df_long

    # ------------------------------------------------------------------
    # 5. Merge all indicators into one panel
    # ------------------------------------------------------------------
    def process_all(self):
        merged = None
        for f in self.files:
            print(f"\n文件: {f}")
            filepath = os.path.join(self.folder_path, f)
            df, note_lines, enc = self.read_and_parse_csv(filepath)
            if df is None:
                print("  读取失败")
                continue

            print(f"  读取成功, 形状: {df.shape}")
            indicator = self.extract_indicator_from_note(note_lines, f)
            print(f"  识别为: {indicator}")

            df_long = self.wide_to_long(df, indicator)
            if df_long is None:
                print("  转置失败")
                continue

            print(f"  转置成功: {len(df_long)} 条记录")

            merged = (
                df_long
                if merged is None
                else pd.merge(merged, df_long, on=["地区", "年份"], how="outer")
            )
            self.indicator_map[f] = indicator

        self.merged_data = merged
        for f, indicator in self.indicator_map.items():
            print(f"  {f} → {indicator}")
        return merged

    # ------------------------------------------------------------------
    # 6. Cleaning: duplicates, outliers, domain validation, interpolation
    # ------------------------------------------------------------------
    def clean_data(self, df):
        df = df.copy()
        original_rows = len(df)
        self.cleaning_report["原始行数"] = original_rows

        before = len(df)
        df = df.drop_duplicates()
        self.cleaning_report["删除重复行"] = before - len(df)
        print(f"  删除重复行: {before - len(df)} 条")

        indicator_cols = [c for c in df.columns if c not in ["地区", "年份"]]
        numeric_cols = [c for c in indicator_cols if df[c].dtype in ["float64", "int64"]]

        outlier_counts = {}
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                before = len(df)
                df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]
                outlier_counts[col] = before - len(df)
                if outlier_counts[col] > 0:
                    print(
                        f"  {col}: 删除异常值 {outlier_counts[col]} 条 "
                        f"(均值={mean:.2f}, 标准差={std:.2f})"
                    )
        self.cleaning_report["异常值删除"] = outlier_counts

        before = len(df)
        if "GDP_亿元" in df.columns:
            df = df[df["GDP_亿元"] > 0]
        if "人口_万人" in df.columns:
            df = df[df["人口_万人"] > 0]
        self.cleaning_report["数据验证删除"] = before - len(df)
        print(f"  数据验证: 删除无效数据 {before - len(df)} 条")

        cols_to_drop = []
        for col in numeric_cols:
            if col in df.columns:
                missing_rate = df[col].isnull().sum() / len(df)
                if missing_rate > 0.5:
                    cols_to_drop.append(col)
                    print(f"  {col}: 缺失率 {missing_rate:.1%} > 50%，删除该列")
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        print("  剩余缺失值处理: 线性插值")
        df = df.sort_values(["地区", "年份"])
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df.groupby("地区")[col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both")
                )

        self.cleaning_report["最终行数"] = len(df)
        self.cleaning_report["删除总数"] = original_rows - len(df)
        print(f"  原始: {original_rows} 条 → 清洗后: {len(df)} 条 "
              f"(删除 {100 * (original_rows - len(df)) / original_rows:.1f}%)")

        self._write_cleaning_report()
        return df

    def _write_cleaning_report(self):
        report_path = OUTPUT_DIR / "cleaning_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"原始记录数: {self.cleaning_report['原始行数']}\n")
            f.write(f"最终记录数: {self.cleaning_report['最终行数']}\n")
            f.write(f"删除记录数: {self.cleaning_report['删除总数']}\n")
            f.write(
                f"删除比例: "
                f"{100 * self.cleaning_report['删除总数'] / max(1, self.cleaning_report['原始行数']):.1f}%\n"
            )
            f.write("\n异常值删除:\n")
            for col, count in self.cleaning_report.get("异常值删除", {}).items():
                f.write(f"  {col}: {count} 条\n")
            f.write(f"\n数据验证删除: {self.cleaning_report.get('数据验证删除', 0)} 条\n")
            f.write(f"重复行删除: {self.cleaning_report.get('删除重复行', 0)} 条\n")
        print(f"\n  清洗报告已保存: {report_path}")

    # ------------------------------------------------------------------
    # 7. Regression + visualization
    # ------------------------------------------------------------------
    def run_regression(self, df):
        indicator_cols = [c for c in df.columns if c not in ["地区", "年份"]]
        if len(indicator_cols) < 2:
            print("指标少于2个，跳过回归")
            return None

        y_col = indicator_cols[0]
        x_cols = indicator_cols[1 : min(3, len(indicator_cols))]

        df_reg = df.dropna(subset=[y_col] + x_cols)
        if len(df_reg) < 10:
            print(f"样本量不足 ({len(df_reg)}条)")
            return None

        y = df_reg[y_col]
        X = sm.add_constant(df_reg[x_cols])
        model = sm.OLS(y, X).fit()

        print(f"因变量: {y_col} | 自变量: {x_cols} | 样本量: {len(df_reg)}")
        print(model.summary())

        results_path = OUTPUT_DIR / "regression_results.txt"
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(f"因变量: {y_col}\n自变量: {x_cols}\n样本量: {len(df_reg)}\n")
            f.write("=" * 60 + "\n\n")
            f.write(model.summary().as_text())
        print(f"\n回归结果已保存: {results_path}")
        return model

    def generate_plots(self, df, model):
        indicator_cols = [c for c in df.columns if c not in ["地区", "年份"]]
        df = df.copy()
        df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
        df = df.dropna(subset=["年份"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if indicator_cols:
            df_grouped = df.groupby("年份")[indicator_cols].mean().reset_index()
            for col in indicator_cols:
                axes[0].plot(df_grouped["年份"], df_grouped[col], marker="o", linewidth=2, label=col)
            axes[0].set_title("各指标全国均值趋势", fontsize=12)
            axes[0].set_xlabel("年份")
            axes[0].set_ylabel("均值")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xticks(sorted(df["年份"].unique()))

        if model is not None and hasattr(model, "params"):
            coefs = model.params.drop("const")
            if len(coefs) > 0:
                axes[1].barh(coefs.index, coefs.values, color="steelblue")
                axes[1].set_title("回归系数", fontsize=12)
                axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5)
                axes[1].grid(True, alpha=0.3)
                for i, v in enumerate(coefs.values):
                    axes[1].text(v + 0.1, i, f"{v:.2f}", va="center")

        plt.tight_layout()
        plot_path = OUTPUT_DIR / "panel_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"图表已保存: {plot_path}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def clean_and_analyze(self):
        if self.merged_data is None:
            print("请先运行 process_all()")
            return None

        cleaned_df = self.clean_data(self.merged_data)
        cleaned_df.to_csv(PROCESSED_DIR / "panel_data.csv", index=False, encoding="utf-8-sig")

        model = self.run_regression(cleaned_df)
        self.generate_plots(cleaned_df, model)

        print("\n生成的文件:")
        print(f"  {PROCESSED_DIR / 'panel_data.csv'} - 清洗后的面板数据")
        print(f"  {OUTPUT_DIR / 'cleaning_report.txt'} - 清洗报告")
        print(f"  {OUTPUT_DIR / 'regression_results.txt'} - 回归结果")
        print(f"  {OUTPUT_DIR / 'panel_analysis.png'} - 趋势图+回归系数图")
        return cleaned_df


if __name__ == "__main__":
    detective = DataDetective(str(RAW_DIR))
    detective.scan_files()
    detective.process_all()
    detective.clean_and_analyze()
