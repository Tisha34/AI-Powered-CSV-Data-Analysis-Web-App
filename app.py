from flask import Flask, render_template, request, jsonify, session
import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from dotenv import load_dotenv
from groq import Groq

app = Flask(__name__)
app.secret_key = os.urandom(24)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"

uploaded_df = None
uploaded_filename = ""


# ─────────────────────────────────────────────
# ROUTE: Home
# ─────────────────────────────────────────────
@app.route("/")
def index():
    session.clear()
    return render_template("index.html")


# ─────────────────────────────────────────────
# ROUTE: Upload CSV
# ─────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_df, uploaded_filename

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    try:
        uploaded_df = pd.read_csv(file)
        uploaded_filename = file.filename

        summary = {
            "filename": uploaded_filename,
            "rows": int(uploaded_df.shape[0]),
            "columns": int(uploaded_df.shape[1]),
            "column_names": uploaded_df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in uploaded_df.dtypes.items()},
            "null_counts": uploaded_df.isnull().sum().to_dict(),
            "preview": uploaded_df.head(5).to_dict(orient="records"),
            "stats": json.loads(
                uploaded_df.describe(include="all").fillna("").to_json()
            ),
        }

        session["history"] = []
        return jsonify({"success": True, "summary": summary}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 500


# ─────────────────────────────────────────────
# ROUTE: Ask question
# ─────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
def ask():
    global uploaded_df

    question = request.form.get("question", "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if uploaded_df is not None:
        data_context = f"""
You are an expert data analyst assistant. The user has uploaded a CSV file called '{uploaded_filename}'.

Dataset overview:
- Shape: {uploaded_df.shape[0]} rows x {uploaded_df.shape[1]} columns
- Columns: {', '.join(uploaded_df.columns.tolist())}
- Data types: {uploaded_df.dtypes.to_dict()}
- Null counts: {uploaded_df.isnull().sum().to_dict()}

First 5 rows:
{uploaded_df.head(5).to_string()}

Summary statistics:
{uploaded_df.describe(include='all').to_string()}

"Answer the user's question based on this data in 2-3 sentences maximum. Give only the direct answer with specific numbers. No markdown, no bold text, no tables, no code blocks, no explanation of how you calculated it. Plain conversational English only."
"""
    else:
        data_context = "You are a helpful personal assistant and data analyst. Answer clearly and concisely."

    history = session.get("history", [])
    history.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=800,
            messages=[{"role": "system", "content": data_context}] + history
        )

        answer = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": answer})
        session["history"] = history

        return jsonify({"response": answer}), 200

    except Exception as e:
        return jsonify({"error": f"API error: {str(e)}"}), 500


# ─────────────────────────────────────────────
# ROUTE: Suggest chart questions from columns
# ─────────────────────────────────────────────
@app.route("/suggest-charts", methods=["POST"])
def suggest_charts():
    global uploaded_df, uploaded_filename

    if uploaded_df is None:
        return jsonify({"error": "Please upload a CSV file first."}), 400

    try:
        numeric_cols = uploaded_df.select_dtypes(include="number").columns.tolist()
        cat_cols = uploaded_df.select_dtypes(include=["object", "category"]).columns.tolist()

        prompt = f"""
You are a data analyst. A user has uploaded a CSV called '{uploaded_filename}'.

Column details:
- Numeric columns: {numeric_cols}
- Categorical columns: {cat_cols}
- First 3 rows: {uploaded_df.head(3).to_dict()}

Generate exactly 5 specific, meaningful chart suggestions for this dataset.
Rules:
- Each suggestion must be a plain English chart request using ACTUAL column names from the data
- Vary the chart types: use bar chart, histogram, line chart, box plot, scatter plot
- Make each suggestion actionable and specific, not generic
- Output ONLY a JSON array of 5 strings, nothing else
- No markdown, no explanation, no preamble
- Example format: ["Bar chart of average Balance by Geography", "Histogram of CreditScore distribution"]
"""

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a data visualization expert. Output only valid JSON arrays, nothing else."},
                {"role": "user", "content": prompt}
            ]
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = "\n".join(raw.split("\n")[:-1])

        suggestions = json.loads(raw)
        if not isinstance(suggestions, list):
            raise ValueError("Response is not a list")
        suggestions = [str(s) for s in suggestions[:5]]

        return jsonify({"suggestions": suggestions}), 200

    except Exception as e:
        return jsonify({"error": f"Could not generate suggestions: {str(e)}"}), 500


# ─────────────────────────────────────────────
# ROUTE: Generate Chart from natural language
# ─────────────────────────────────────────────
@app.route("/chart", methods=["POST"])
def chart():
    global uploaded_df

    if uploaded_df is None:
        return jsonify({"error": "Please upload a CSV file first."}), 400

    chart_request = request.form.get("chart_request", "").strip()
    if not chart_request:
        return jsonify({"error": "Please describe the chart you want."}), 400

    code_prompt = f"""The user has a pandas DataFrame called `df` with these columns: {uploaded_df.columns.tolist()}
Dtypes: {uploaded_df.dtypes.to_dict()}
First 3 rows: {uploaded_df.head(3).to_dict()}

The user wants: "{chart_request}"

Write ONLY executable Python matplotlib code. Follow every rule below exactly:

RULES:
1. Always start with: fig, ax = plt.subplots(figsize=(8, 4))
2. Do NOT call plt.show()
3. Do NOT import anything — pandas and matplotlib are already imported
4. Do NOT include comments, markdown, explanation, or backticks
5. Always close ALL parentheses, brackets, and braces before the code ends
6. For grouped comparisons like "exited and stayed by gender" or "churn by category":
   - Use grouped bar charts with ax.bar() called multiple times with different x offsets
   - Example for two groups:
     x = np.arange(len(categories))
     ax.bar(x - 0.2, values1, width=0.4, label='Group1', color='#a8d8ea')
     ax.bar(x + 0.2, values2, width=0.4, label='Group2', color='#f9c8d4')
     ax.set_xticks(x)
     ax.set_xticklabels(categories)
     ax.legend()
7. For single category bar charts, color each bar differently:
   colors = ['#a8d8ea', '#f9c8d4', '#b8e0d2', '#ffd3a3', '#c9b8e8', '#ffe0a3']
   bars = ax.bar(x_values, y_values, color=colors[:len(x_values)])
8. Always add: ax.set_title('...'), ax.set_xlabel('...'), ax.set_ylabel('...')
9. If the request is unclear, make a reasonable interpretation using actual column names
10. End with plt.tight_layout()
"""

    try:
        code_response = client.chat.completions.create(
            model=MODEL,
            max_tokens=600,
            messages=[
                {"role": "system", "content": "You are a Python data visualization expert. Output only raw executable Python code, nothing else."},
                {"role": "user", "content": code_prompt}
            ]
        )

        code = code_response.choices[0].message.content.strip()
        if code.startswith("```"):
            code = "\n".join(code.split("\n")[1:])
        if code.endswith("```"):
            code = "\n".join(code.split("\n")[:-1])

        local_scope = {"df": uploaded_df, "plt": plt, "pd": pd}
        plt.rcParams.update({
            'figure.facecolor': '#1a1d27',
            'axes.facecolor': '#1a1d27',
            'axes.edgecolor': '#2e3347',
            'axes.labelcolor': '#c9d1e0',
            'xtick.color': '#c9d1e0',
            'ytick.color': '#c9d1e0',
            'text.color': '#c9d1e0',
            'grid.color': '#2e3347',
            'grid.linewidth': 0.8,
            'axes.prop_cycle': plt.cycler(color=[
                '#a8d8ea', '#b8e0d2', '#f9c8d4',
                '#ffd3a3', '#c9b8e8', '#ffe0a3'
            ])
        })
        exec(code, {}, local_scope)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify({"chart": img_base64}), 200

    except Exception as e:
        plt.close("all")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# ROUTE: Data Quality Scorecard
# ─────────────────────────────────────────────
@app.route("/data-quality", methods=["POST"])
def data_quality():
    global uploaded_df, uploaded_filename

    if uploaded_df is None:
        return jsonify({"error": "Please upload a CSV file first."}), 400

    try:
        df = uploaded_df
        total_cells = df.shape[0] * df.shape[1]
        issues = []
        penalties = 0

        # ── 1. Missing values ──────────────────────
        missing_total = int(df.isnull().sum().sum())
        missing_pct = round(missing_total / total_cells * 100, 2)
        missing_cols = [
            {
                "column": col,
                "missing": int(df[col].isnull().sum()),
                "pct": round(df[col].isnull().mean() * 100, 2),
                "recommendation": "Drop column" if df[col].isnull().mean() > 0.5
                                  else "Impute with median" if df[col].dtype in ["float64", "int64"]
                                  else "Impute with mode"
            }
            for col in df.columns if df[col].isnull().sum() > 0
        ]
        if missing_pct > 20:
            penalties += 25
            issues.append(f"High missing data: {missing_pct}% of values are null")
        elif missing_pct > 5:
            penalties += 12
            issues.append(f"Moderate missing data: {missing_pct}% of values are null")
        elif missing_pct > 0:
            penalties += 5
            issues.append(f"Minor missing data: {missing_pct}% of values are null")

        # ── 2. Duplicate rows ──────────────────────
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = round(duplicate_count / df.shape[0] * 100, 2)
        if duplicate_pct > 10:
            penalties += 20
            issues.append(f"High duplicate rows: {duplicate_count} rows ({duplicate_pct}%)")
        elif duplicate_pct > 2:
            penalties += 10
            issues.append(f"Some duplicate rows: {duplicate_count} rows ({duplicate_pct}%)")
        elif duplicate_count > 0:
            penalties += 3
            issues.append(f"Minor duplicates: {duplicate_count} rows ({duplicate_pct}%)")

        # ── 3. Outliers (IQR method) ───────────────
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        outlier_cols = []
        total_outliers = 0
        for col in numeric_cols:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if outlier_count > 0:
                outlier_pct = round(outlier_count / len(s) * 100, 2)
                total_outliers += outlier_count
                outlier_cols.append({
                    "column": col,
                    "count": outlier_count,
                    "pct": outlier_pct,
                    "recommendation": "Investigate and cap outliers" if outlier_pct > 5
                                      else "Review individual records"
                })
                if outlier_pct > 10:
                    penalties += 10
                elif outlier_pct > 5:
                    penalties += 5
                else:
                    penalties += 2

        if total_outliers > 0:
            issues.append(f"Outliers detected in {len(outlier_cols)} column(s) — {total_outliers} total records affected")

        # ── 4. Constant / near-constant columns ───
        constant_cols = []
        for col in df.columns:
            unique_ratio = df[col].nunique() / df.shape[0]
            if unique_ratio < 0.01 and df[col].nunique() == 1:
                constant_cols.append(col)
                penalties += 5
                issues.append(f"Column '{col}' has only one unique value — provides no analytical value")
            elif unique_ratio < 0.01 and df[col].nunique() > 1:
                constant_cols.append(col)
                penalties += 2

        # ── 5. High cardinality categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        high_cardinality = []
        for col in cat_cols:
            unique_ratio = df[col].nunique() / df.shape[0]
            if unique_ratio > 0.9:
                high_cardinality.append(col)
                penalties += 3
                issues.append(f"Column '{col}' has very high cardinality ({df[col].nunique()} unique values) — may be an ID column")

        # ── 6. Mixed data types detection ─────────
        mixed_cols = []
        for col in df.select_dtypes(include="object").columns:
            sample = df[col].dropna().head(100)
            numeric_looking = sample.str.match(r"^-?\d+\.?\d*$", na=False).sum()
            if numeric_looking > len(sample) * 0.5:
                mixed_cols.append(col)
                penalties += 8
                issues.append(f"Column '{col}' appears numeric but stored as text — consider converting dtype")

        # ── 7. Compute final score ─────────────────
        score = max(0, 100 - penalties)

        if score >= 85:
            verdict = "analysis-ready"
            verdict_detail = "This dataset is in good shape and ready for analysis."
            verdict_color = "good"
        elif score >= 65:
            verdict = "needs minor cleaning"
            verdict_detail = "This dataset has some issues that should be addressed before analysis."
            verdict_color = "warn"
        else:
            verdict = "needs significant cleaning"
            verdict_detail = "This dataset has serious quality issues. Clean it before drawing conclusions."
            verdict_color = "bad"

        # ── 8. LLM recommendations ─────────────────
        rec_prompt = f"""
You are a senior data analyst reviewing a dataset called '{uploaded_filename}'.

Data Quality Score: {score}/100
Verdict: {verdict}

Issues found:
{chr(10).join(f'- {i}' for i in issues) if issues else '- No major issues found'}

Missing value columns: {[c['column'] + ' (' + str(c['pct']) + '%)' for c in missing_cols]}
Outlier columns: {[c['column'] + ' (' + str(c['pct']) + '% outliers)' for c in outlier_cols]}
High cardinality columns: {high_cardinality}
Mixed type columns: {mixed_cols}

Write exactly 3-4 specific, actionable cleaning recommendations for this dataset.
Rules:
- Each recommendation must reference actual column names from the data
- Be specific about what action to take and why
- Write in plain English, no bullet point symbols, just numbered lines like 1. 2. 3.
- Keep each recommendation to one sentence
- Do not repeat the issues, just give the fix
"""

        rec_response = client.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a senior data analyst giving concise data cleaning advice."},
                {"role": "user", "content": rec_prompt}
            ]
        )
        recommendations = rec_response.choices[0].message.content.strip()

        result = {
            "score": score,
            "verdict": verdict,
            "verdict_detail": verdict_detail,
            "verdict_color": verdict_color,
            "issues": issues,
            "missing_cols": missing_cols,
            "duplicate_count": duplicate_count,
            "duplicate_pct": duplicate_pct,
            "outlier_cols": outlier_cols,
            "high_cardinality": high_cardinality,
            "mixed_cols": mixed_cols,
            "recommendations": recommendations,
            "summary": {
                "total_cells": total_cells,
                "missing_pct": missing_pct,
                "duplicate_pct": duplicate_pct,
                "outlier_columns": len(outlier_cols),
            }
        }

        return jsonify({"success": True, "result": result}), 200

    except Exception as e:
        return jsonify({"error": f"Data quality check failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# ROUTE: Auto EDA Report
# ─────────────────────────────────────────────
@app.route("/eda", methods=["POST"])
def eda():
    global uploaded_df, uploaded_filename

    if uploaded_df is None:
        return jsonify({"error": "Please upload a CSV file first."}), 400

    try:
        df = uploaded_df
        report = {}

        # ── 1. Shape & basic info ──────────────────
        report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

        # ── 2. Missing values ──────────────────────
        null_counts = df.isnull().sum()
        null_pct = (df.isnull().mean() * 100).round(2)
        report["missing"] = [
            {"column": col, "missing": int(null_counts[col]), "pct": float(null_pct[col])}
            for col in df.columns if null_counts[col] > 0
        ]

        # ── 3. Numeric column stats + outliers ────
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        numeric_stats = []
        outlier_info = []

        for col in numeric_cols:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            skew = float(round(s.skew(), 3))

            numeric_stats.append({
                "column": col,
                "mean": float(round(s.mean(), 3)),
                "median": float(round(s.median(), 3)),
                "std": float(round(s.std(), 3)),
                "min": float(s.min()),
                "max": float(s.max()),
                "skew": skew,
                "skew_flag": "right-skewed" if skew > 1 else "left-skewed" if skew < -1 else "normal",
            })

            if outliers > 0:
                outlier_info.append({
                    "column": col,
                    "outlier_count": outliers,
                    "pct": float(round(outliers / len(s) * 100, 2))
                })

        report["numeric_stats"] = numeric_stats
        report["outliers"] = outlier_info

        # ── 4. Categorical columns ─────────────────
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_stats = []
        for col in cat_cols:
            vc = df[col].value_counts()
            cat_stats.append({
                "column": col,
                "unique": int(df[col].nunique()),
                "top_value": str(vc.index[0]) if len(vc) > 0 else "",
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
            })
        report["categorical_stats"] = cat_stats

        # ── 5. Top correlations ────────────────────
        top_corr = []
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().abs()
            mask = np.tril(np.ones(corr_matrix.shape)).astype(bool)
            pairs = corr_matrix.where(~mask).stack().reset_index()
            pairs.columns = ["col1", "col2", "correlation"]
            pairs = pairs.sort_values("correlation", ascending=False).head(5)
            top_corr = [
                {"col1": r["col1"], "col2": r["col2"],
                 "correlation": float(round(r["correlation"], 3))}
                for _, r in pairs.iterrows()
            ]
        report["top_correlations"] = top_corr

        # ── 6. LLM narrative ──────────────────────
        eda_summary_prompt = f"""
You are a senior data analyst. Based on the following EDA results for a dataset called '{uploaded_filename}', write a concise 4-5 sentence insight narrative.

Dataset shape: {report['shape']['rows']} rows, {report['shape']['columns']} columns
Missing values: {report['missing'] if report['missing'] else 'None'}
Outliers detected: {report['outliers'] if report['outliers'] else 'None'}
Top correlations: {report['top_correlations'] if report['top_correlations'] else 'None'}
Skewed columns: {[s for s in report['numeric_stats'] if s['skew_flag'] != 'normal']}

Instructions:
- Write in plain English, no bullet points
- Mention the most important finding first
- Flag data quality issues if present
- Mention the strongest correlation and what it might mean
- End with one recommendation for next analysis steps
- Do NOT use technical jargon
"""

        narrative_response = client.chat.completions.create(
            model=MODEL,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a senior data analyst writing executive-level data summaries."},
                {"role": "user", "content": eda_summary_prompt}
            ]
        )
        report["narrative"] = narrative_response.choices[0].message.content.strip()

        # ── 7. Correlation heatmap ─────────────────
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(8, 5))
            corr_data = df[numeric_cols].corr()
            im = ax.imshow(corr_data.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(numeric_cols, fontsize=9)
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax.text(j,i,f"{corr_data.values[i,j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(corr_data.values[i,j]) > 0.6 else "black")
            plt.colorbar(im, ax=ax)
            ax.set_title("Correlation Heatmap", fontsize=12, pad=12)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            plt.close("all")
            buf.seek(0)
            report["heatmap"] = base64.b64encode(buf.read()).decode("utf-8")
        else:
            report["heatmap"] = None

        return jsonify({"success": True, "report": report}), 200

    except Exception as e:
        plt.close("all")
        return jsonify({"error": f"EDA failed: {str(e)}"}), 500


# ─────────────────────────────────────────────
# ROUTE: Clear conversation history
# ─────────────────────────────────────────────
@app.route("/clear", methods=["POST"])
def clear():
    session["history"] = []
    return jsonify({"success": True}), 200


if __name__ == "__main__":
    app.run(debug=True)
    