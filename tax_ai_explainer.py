#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启贤托育AI税务助手 - 命令行版（支持真实 DeepSeek AI）
运行：python tax_ai_explainer.py

配置 API Key（任选一种）：
  方式1：在项目根目录创建 .env 文件，写入 DEEPSEEK_API_KEY=sk-xxx
  方式2：设置环境变量 DEEPSEEK_API_KEY
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# 读取 .env 文件
load_dotenv()

# ===============================================
#  DeepSeek AI 配置
# ===============================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# 是否使用真实 AI
USE_REAL_AI = bool(DEEPSEEK_API_KEY)

if USE_REAL_AI:
    print("[OK] 检测到 DeepSeek API Key，将使用真实 AI 生成申报说明")
else:
    print("[提示] 未检测到 API Key，使用模拟模式（规则生成）")
    print("       如需使用真实 AI，请在 .env 文件中配置 DEEPSEEK_API_KEY")


# ===============================================
#  社保与个税计算函数（与 tax_calculator.py 保持一致）
# ===============================================

# 武汉 2026 社保比例（个人）
SOCIAL_INSURANCE_RATE_PERSONAL = {
    "养老": 0.08,
    "医疗": 0.02,
    "失业": 0.003,
}
SOCIAL_INSURANCE_PERSONAL_FIXED = {
    "医疗大病": 7,   # 大额医疗保险固定 7 元（鄂人社发〔2023〕）
}

# 公司承担部分
SOCIAL_INSURANCE_RATE_COMPANY = {
    "养老": 0.16,
    "医疗": 0.087,
    "失业": 0.007,
    "工伤": 0.002,
}

def calc_social_insurance_personal(base):
    """计算个人社保缴纳金额"""
    total = 0
    detail = {}
    for k, v in SOCIAL_INSURANCE_RATE_PERSONAL.items():
        amount = base * v
        detail[k] = round(amount, 2)
        total += amount
    for k, v in SOCIAL_INSURANCE_PERSONAL_FIXED.items():
        detail[k] = v
        total += v
    return round(total, 2), detail

def calc_social_insurance_company(base):
    """计算公司承担社保金额"""
    total = 0
    detail = {}
    for k, v in SOCIAL_INSURANCE_RATE_COMPANY.items():
        amount = base * v
        detail[k] = round(amount, 2)
        total += amount
    return round(total, 2), detail

def calc_income_tax(taxable_income):
    """计算个税（累计预扣法，此处简化为月度计算）"""
    if taxable_income <= 0:
        return 0, 0.0
    brackets = [
        (36000,    0.03, 0),
        (144000,   0.10, 2520),
        (300000,   0.20, 16920),
        (420000,   0.25, 31920),
        (660000,   0.30, 52920),
        (960000,   0.35, 85920),
        (float('inf'), 0.45, 181920),
    ]
    remaining = taxable_income
    tax = 0
    for threshold, rate, deduction in brackets:
        if remaining <= threshold:
            tax = remaining * rate - deduction
            break
        remaining = threshold
    # 注意：以上为年度累计逻辑简化，实际月度申报用累计预扣法
    # 此处做简化演示，建议对接专业税务计算库
    return max(0, round(tax, 2)), 0.0

def calc_one_employee(name, gross_salary, si_base, si_personal_actual,
                     special_deduction, child_edu=0, infant_care=0, elderly_care=0):
    """计算单名员工税务详情"""
    # 如果用实际缴纳金额，以实际为准；否则按基数计算
    if si_personal_actual > 0:
        si_personal = si_personal_actual
    else:
        si_personal, _ = calc_social_insurance_personal(si_base)

    taxable = gross_salary - si_personal - 5000 - special_deduction
    tax, _ = calc_income_tax(max(0, taxable))

    net_salary = gross_salary - si_personal - tax

    company_si, company_si_detail = calc_social_insurance_company(si_base)
    total_labor_cost = gross_salary + company_si

    return {
        "姓名": name,
        "税前工资": gross_salary,
        "个人社保": si_personal,
        "专项附加扣除": special_deduction,
        "子女教育": child_edu,
        "婴幼儿照护": infant_care,
        "赡养老人": elderly_care,
        "应税收入": max(0, round(taxable, 2)),
        "应纳税额": tax,
        "实发工资": round(net_salary, 2),
        "公司社保承担": company_si,
        "公司用人总成本": round(total_labor_cost, 2),
        "公司社保明细": company_si_detail,
    }


# ===============================================
#  AI 申报说明生成
# ===============================================

def generate_ai_explanation(results, year, month):
    """调用 DeepSeek API 生成专业申报说明"""

    # 构造提示词
    rows_text = ""
    for r in results:
        rows_text += (
            f"员工 {r['姓名']}：税前工资 {r['税前工资']} 元，"
            f"个人社保 {r['个人社保']} 元，"
            f"专项附加扣除 {r['专项附加扣除']} 元"
            f"（子女教育 {r['子女教育']} 元，婴幼儿照护 {r['婴幼儿照护']} 元，赡养老人 {r['赡养老人']} 元），"
            f"应税收入 {r['应税收入']} 元，应纳税额 {r['应纳税额']} 元，"
            f"实发工资 {r['实发工资']} 元。\n"
        )

    company_si_total = sum(r["公司社保承担"] for r in results)
    total_labor_cost = sum(r["公司用人总成本"] for r in results)
    total_tax = sum(r["应纳税额"] for r in results)

    prompt = f"""你是一位专业的税务顾问，请为以下企业{year}年{month}月的个税及社保申报撰写一份专业的申报说明。

## 员工数据
{rows_text}
## 汇总数据
- 公司承担社保总额：{company_si_total} 元
- 全体员工应纳税额合计：{total_tax} 元
- 公司用人总成本：{total_labor_cost} 元

## 要求
1. 以"湖北启贤托儿所有限公司 {year}年{month}月 税务申报说明"为标题
2. 分四个部分：一、申报概况；二、员工个税明细；三、社保缴纳说明；四、申报注意事项
3. 语气专业、简洁，适合财务提交给税务局或留存备案
4. 提醒用户核对专项附加扣除信息是否已及时更新（个税APP）
5. 说明社保基数如有调整请以社保局核定为准
6. 总字数控制在 500-800 字
7. 用中文输出，不要输出英文
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一位专业的税务顾问，擅长撰写企业税务申报说明。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.3,
    }

    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"[AI调用失败 {resp.status_code}]，切换模拟模式")
            return generate_mock_explanation(results, year, month)
    except Exception as e:
        print(f"[AI调用异常: {e}]，切换模拟模式")
        return generate_mock_explanation(results, year, month)


def generate_mock_explanation(results, year, month):
    """模拟 AI 生成申报说明（无 API Key 时的降级方案）"""
    lines = []
    lines.append(f"湖北启贤托儿所有限公司 {year}年{month}月 税务申报说明\n")
    lines.append("=" * 50)
    lines.append("\n一、申报概况\n")
    lines.append(f"  本月申报员工人数：{len(results)} 人")
    total_tax = sum(r["应纳税额"] for r in results)
    company_si = sum(r["公司社保承担"] for r in results)
    lines.append(f"  应纳税额合计：{total_tax} 元")
    lines.append(f"  公司承担社保合计：{company_si} 元")
    lines.append("\n二、员工个税明细\n")
    for r in results:
        lines.append(f"  {r['姓名']}：应税收入 {r['应税收入']} 元，应纳税额 {r['应纳税额']} 元，实发 {r['实发工资']} 元")
    lines.append("\n三、社保缴纳说明\n")
    lines.append(f"  社保缴费基数：{results[0]['个人社保']} 元（以实际申报为准）")
    lines.append(f"  公司承担部分合计：{company_si} 元")
    lines.append("\n四、申报注意事项\n")
    lines.append("  1. 请核对专项附加扣除信息是否已及时更新（个税APP）")
    lines.append("  2. 社保基数如有调整请以社保局核定为准")
    lines.append("  3. 本底稿由 AI 辅助生成，提交前请人工复核")
    lines.append("\n" + "=" * 50 + "\n")
    return "\n".join(lines)


def save_results(results, explanation, year, month):
    """保存计算结果和申报说明"""
    # 保存 CSV 底稿
    import csv
    csv_path = f"申报底稿_{year}{month:02d}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] CSV底稿已生成：{csv_path}")

    # 保存申报说明
    txt_path = f"申报说明_{year}{month:02d}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(explanation)
    print(f"[OK] 申报说明已生成：{txt_path}")

    return csv_path, txt_path


# ===============================================
#  主程序
# ===============================================

if __name__ == "__main__":
    now = datetime.now()
    year, month = now.year, now.month

    print(f"\n启贤托育AI税务助手 - {year}年{month}月申报\n")
    print("-" * 50)

    # 员工数据（可修改为 Excel 导入）
    employees = [
        {
            "name": "员工A",
            "gross": 10522,
            "si_base": 5000,
            "si_actual": 522,
            "special": 5000,
            "child_edu": 2000,
            "infant": 2000,
            "elderly": 1000,
        },
    ]

    results = []
    for emp in employees:
        r = calc_one_employee(
            emp["name"], emp["gross"], emp["si_base"], emp["si_actual"],
            emp["special"], emp["child_edu"], emp["infant"], emp["elderly"]
        )
        results.append(r)
        print(f"  {r['姓名']}：个税 {r['应纳税额']} 元，实发 {r['实发工资']} 元")

    print("\n正在生成 AI 申报说明...\n")
    explanation = generate_ai_explanation(results, year, month)
    print(explanation)

    save_results(results, explanation, year, month)

    print("\n[提示] 使用说明：")
    print("  1. 修改上方 employees 列表增加员工")
    print("  2. 或接入 Excel 导入（参考 tax_web_app.py）")
    print("  3. 确保 .env 文件已配置 DEEPSEEK_API_KEY 以使用真实 AI\n")
