#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
湖北启贤托儿所有限公司（启贤托育）- 个税/社保计算脚本（MVP版）
适用场景：零申报企业，有工资发放和社保缴纳
运行：python tax_calculator.py
"""

from datetime import datetime
import json
import os

# ===============================================
#  政策参数加载器（从 tax_policies.json 读取）
# ===============================================

_POLICIES_CACHE = None
_POLICIES_CACHE_TIME = None
_POLICIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tax_policies.json")


def _load_raw_policies():
    """加载原始 JSON（带缓存，60秒刷新）"""
    global _POLICIES_CACHE, _POLICIES_CACHE_TIME
    now = datetime.now()
    if _POLICIES_CACHE is not None and _POLICIES_CACHE_TIME is not None:
        if (now - _POLICIES_CACHE_TIME).seconds < 60:
            return _POLICIES_CACHE
    with open(_POLICIES_PATH, "r", encoding="utf-8") as f:
        _POLICIES_CACHE = json.load(f)
    _POLICIES_CACHE_TIME = now
    return _POLICIES_CACHE


def load_tax_policies(year: int = None):
    """
    加载指定年度的税收政策参数。

    查找逻辑：
    1. 按 year 匹配 policy_periods 中的区间
    2. 如果匹配到的区间 status="placeholder" 且有 inherit_from，回退到源区间
    3. 返回一个扁平化的参数字典，供各计算函数使用

    参数：
      year: 申报年度，默认当前年
    返回：
      dict 包含所有税种参数 + 元数据
    """
    if year is None:
        year = datetime.now().year

    raw = _load_raw_policies()
    periods = raw.get("policy_periods", [])

    # 1. 查找匹配区间
    matched = None
    for p in periods:
        start = int(p["effective_from"][:4])
        end = int(p["effective_until"][:4])
        if start <= year <= end:
            matched = p
            break

    if matched is None:
        # 超出所有区间范围，使用最后一个
        matched = periods[-1] if periods else {}

    # 2. 如果是占位区间，回退到继承源
    if matched.get("status") == "placeholder" and matched.get("inherit_from"):
        inherit_period = matched["inherit_from"]
        for p in periods:
            if p["period"] == inherit_period:
                # 深度合并：占位区间覆盖源区间（占位区间可能有部分覆盖值）
                matched = _deep_merge(p.copy(), matched)
                break

    # 3. 提取特殊附加扣除（跨税种共用）
    special_deductions = raw.get("special_deductions", {})

    return {
        "_meta": {
            "period": matched.get("period", "unknown"),
            "status": matched.get("status", "unknown"),
            "label": matched.get("label", ""),
            "summary": matched.get("summary", ""),
            "effective_from": matched.get("effective_from", ""),
            "effective_until": matched.get("effective_until", ""),
            "policy_version": raw.get("meta", {}).get("version", ""),
        },
        "personal_income_tax": matched.get("personal_income_tax", {}),
        "social_insurance": matched.get("social_insurance", {}),
        "vat": matched.get("vat", {}),
        "surcharges": matched.get("surcharges", {}),
        "corporate_income_tax": matched.get("corporate_income_tax", {}),
        "stamp_duty": matched.get("stamp_duty", {}),
        "disabled_employment_fund": matched.get("disabled_employment_fund", {}),
        "special_deductions": special_deductions,
        "auto_update": raw.get("auto_update", {}),
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并：override 中的值覆盖 base（但保留 base 中 override 没有的键）"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ===============================================
#  向后兼容：保留模块级常量（从 JSON 默认加载）
# ===============================================

_default_pol = load_tax_policies()

SOCIAL_INSURANCE = {
    "pension_personal": _default_pol["social_insurance"]["personal_rates"]["pension"],
    "medical_personal": _default_pol["social_insurance"]["personal_rates"]["medical"],
    "unemployment_personal": _default_pol["social_insurance"]["personal_rates"]["unemployment"],
}

SOCIAL_INSURANCE_ACTUAL = _default_pol["social_insurance"]["personal_actual"]  # 522 = 基数×10.3%+大病7元

SOCIAL_INSURANCE_COMPANY = {
    "pension": _default_pol["social_insurance"]["company_rates"]["pension"],
    "medical": _default_pol["social_insurance"]["company_rates"]["medical"],
    "unemployment": _default_pol["social_insurance"]["company_rates"]["unemployment"],
    "injury": _default_pol["social_insurance"]["company_rates"]["injury"],
}

CRITICAL_ILLNESS_FIXED = _default_pol["social_insurance"].get("critical_illness_fixed", 7)

HOUSING_FUND_PERSONAL_RATE = _default_pol["social_insurance"]["housing_fund_personal"]
HOUSING_FUND_COMPANY_RATE = _default_pol["social_insurance"]["housing_fund_company"]

TAX_BRACKETS = [
    tuple(b) for b in _default_pol["personal_income_tax"]["brackets"]
]

BASIC_DEDUCTION = _default_pol["personal_income_tax"]["basic_deduction"]

# ===============================================


def calc_social_insurance_company(base=5000):
    """计算公司社保部分（大病医保已含在医疗8.7%费率中，不另计）"""
    return (
        base * SOCIAL_INSURANCE_COMPANY["pension"]
        + base * SOCIAL_INSURANCE_COMPANY["medical"]
        + base * SOCIAL_INSURANCE_COMPANY["unemployment"]
        + base * SOCIAL_INSURANCE_COMPANY["injury"]
    )


def calc_income_tax(
    gross_salary: float,
    social_insurance_personal: float,
    housing_fund_personal: float,
    special_deductions: float,
) -> tuple[float, float]:
    """
    计算个人所得税
    返回：(应纳税额, 应税收入)
    """
    taxable_income = (
        gross_salary
        - social_insurance_personal
        - housing_fund_personal
        - BASIC_DEDUCTION
        - special_deductions
    )

    if taxable_income <= 0:
        return 0.0, 0.0

    # 查找适用税率
    tax_rate = 0.03
    quick_deduction = 0
    for lower, upper, rate, deduction in TAX_BRACKETS:
        if lower < taxable_income <= upper:
            tax_rate = rate
            quick_deduction = deduction
            break

    tax = taxable_income * tax_rate - quick_deduction
    return round(tax, 2), round(taxable_income, 2)


def format_money(val: float) -> str:
    """格式化金额显示"""
    return f"{val:,.2f}"


def process_employees(employees: list[dict]) -> list[dict]:
    """
    处理员工列表，返回计算结果
    employees 每项格式：
    {
        "name": "员工A",
        "gross_salary": 10522,
        "si_base": 5000,
        "si_personal_actual": 522,   # 个人社保实缴 = 基数×(8%+2%+0.3%)+大病7元
        "special_deductions": 5000,   # 专项附加扣除合计
        "child_education": 2000,     # 明细（可选，用于底稿）
        "infant_care": 2000,
        "elderly_care": 1000,
    }
    """
    results = []

    for emp in employees:
        gross = emp["gross_salary"]
        si_base = emp.get("si_base", 5000)
        si_personal = emp.get("si_personal_actual", SOCIAL_INSURANCE_ACTUAL)
        hf_personal = emp.get("gross_salary", 0) * HOUSING_FUND_PERSONAL_RATE
        if emp.get("housing_fund_personal"):
            hf_personal = emp["housing_fund_personal"]
        special = emp.get("special_deductions", 0)

        tax, taxable_income = calc_income_tax(
            gross, si_personal, hf_personal, special
        )

        net_salary = gross - si_personal - hf_personal - tax
        si_company = calc_social_insurance_company(si_base)
        hf_company = si_base * HOUSING_FUND_COMPANY_RATE
        total_cost = gross + si_company + hf_company

        results.append({
            "姓名": emp["name"],
            "税前工资": gross,
            "个人社保": si_personal,
            "个人公积金": hf_personal,
            "专项附加扣除": special,
            "应税收入": taxable_income,
            "应纳税额": tax,
            "实发工资": round(net_salary, 2),
            "公司社保承担": round(si_company, 2),
            "公司公积金承担": round(hf_company, 2),
            "公司用人总成本": round(total_cost, 2),
            # 明细（用于底稿）
            "子女教育": emp.get("child_education", 0),
            "婴幼儿照护": emp.get("infant_care", 0),
            "赡养老人": emp.get("elderly_care", 0),
        })

    return results


def print_results(results: list[dict]):
    """打印计算结果"""
    print("\n" + "=" * 70)
    print("  个税/社保计算结果  |  湖北启贤托儿所有限公司")
    print("=" * 70)

    for r in results:
        print(f"\n【{r['姓名']}】")
        print(f"  税前工资：      {format_money(r['税前工资'])} 元")
        print(f"  个人社保扣款：  {format_money(r['个人社保'])} 元")
        print(f"  专项附加扣除：  {format_money(r['专项附加扣除'])} 元")
        print(f"    ├─ 子女教育： {format_money(r['子女教育'])} 元")
        print(f"    ├─ 婴幼儿照护：{format_money(r['婴幼儿照护'])} 元")
        print(f"    └─ 赡养老人： {format_money(r['赡养老人'])} 元")
        print(f"  应税收入：      {format_money(r['应税收入'])} 元")
        print(f"  应纳税额：      {format_money(r['应纳税额'])} 元")
        print(f"  实发工资：      {format_money(r['实发工资'])} 元")
        print(f"  公司社保承担：  {format_money(r['公司社保承担'])} 元")
        print(f"  公司用人总成本：{format_money(r['公司用人总成本'])} 元")

    print("\n" + "-" * 70)
    print("  【汇总】")
    total_gross = sum(r["税前工资"] for r in results)
    total_tax = sum(r["应纳税额"] for r in results)
    total_net = sum(r["实发工资"] for r in results)
    total_si_company = sum(r["公司社保承担"] for r in results)
    total_cost = sum(r["公司用人总成本"] for r in results)
    print(f"  工资总额：      {format_money(total_gross)} 元")
    print(f"  个税总额：      {format_money(total_tax)} 元")
    print(f"  实发工资总额：  {format_money(total_net)} 元")
    print(f"  公司社保总额：  {format_money(total_si_company)} 元")
    print(f"  公司用人总成本：{format_money(total_cost)} 元")
    print("=" * 70)


def export_csv(results: list[dict], output_path: str = None):
    """导出CSV底稿（无需pandas依赖）"""
    if output_path is None:
        month = datetime.now().strftime("%Y%m")
        output_path = f"申报底稿_{month}.csv"

    # 表头
    headers = [
        "姓名", "税前工资", "个人社保", "个人公积金",
        "专项附加扣除合计", "子女教育", "婴幼儿照护", "赡养老人",
        "应税收入", "应纳税额", "实发工资",
        "公司社保承担", "公司公积金承担", "公司用人总成本"
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                r["姓名"],
                str(r["税前工资"]),
                str(r["个人社保"]),
                str(r["个人公积金"]),
                str(r["专项附加扣除"]),
                str(r["子女教育"]),
                str(r["婴幼儿照护"]),
                str(r["赡养老人"]),
                str(r["应税收入"]),
                str(r["应纳税额"]),
                str(r["实发工资"]),
                str(r["公司社保承担"]),
                str(r["公司公积金承担"]),
                str(r["公司用人总成本"]),
            ]
            f.write(",".join(row) + "\n")

    print(f"\n[OK] CSV底稿已生成：{output_path}")
    return output_path


# ===============================================
#  企业所得税季度预缴（小型微利企业优惠）
# ===============================================

def calc_vat_and_surcharge(
    revenue: float,
    vat_rate: float = 0.03,
    is_small_scale: bool = True,
    is_small_low_profit: bool = True,
    vat_paid_ytd: float = 0.0,
    tax_year: int = None,
) -> dict:
    """
    计算增值税及附加税（城建税、教育费附加、地方教育附加）

    税率参数从 tax_policies.json 读取，按申报年度自动匹配对应政策区间。

    参数：
      revenue:          季度含税营业收入（元）
      vat_rate:         增值税名义税率（小规模原3%，实际减按1%）
      is_small_scale:   是否小规模纳税人
      is_small_low_profit: 是否小型微利企业（影响六税两费减半资格）
      vat_paid_ytd:     本年已缴增值税（用于校验）
      tax_year:         申报年度（默认当前年，用于匹配政策区间）
    """
    pol = load_tax_policies(tax_year if tax_year else None)
    vat_pol = pol["vat"]
    sur_pol = pol["surcharges"]

    # ===== 小规模纳税人：含税收入换算不含税收入 =====
    if is_small_scale:
        revenue_excl = round(revenue / (1 + vat_rate), 2)
    else:
        revenue_excl = revenue

    # ===== 增值税计算 =====
    exempt_threshold = vat_pol["quarterly_exempt_threshold"]
    effective_rate = vat_pol["small_scale_effective_rate"]
    nominal_rate = vat_pol["small_scale_nominal_rate"]
    vat_ref = vat_pol["policy_ref"]

    if is_small_scale and revenue_excl <= exempt_threshold:
        vat = 0.0
        vat_note = f"季度不含税收入 {revenue_excl:,.2f} 元 ≤ {exempt_threshold//10000}万 → 免征增值税"
        vat_policy = f"{vat_ref}：小规模纳税人季度≤{exempt_threshold//10000}万免征增值税"
        vat_effective_rate = 0.0
    elif is_small_scale:
        vat = round(revenue_excl * effective_rate, 2)
        vat_note = f"季度不含税收入 {revenue_excl:,.2f} 元 > {exempt_threshold//10000}万 → 减按{effective_rate*100:.0f}%征收（原{nominal_rate*100:.0f}%）"
        vat_policy = f"{vat_ref}：小规模纳税人减按{effective_rate*100:.0f}%征收"
        vat_effective_rate = effective_rate
    else:
        vat = round(revenue_excl * vat_rate, 2)
        vat_note = f"一般纳税人按{vat_rate*100:.0f}%征收"
        vat_policy = "一般纳税人标准税率"
        vat_effective_rate = vat_rate

    # ===== 六税两费减半判断 =====
    six_two_half = sur_pol.get("six_two_half_enabled", True) and (is_small_scale or is_small_low_profit)
    half = sur_pol["half_multiplier"] if six_two_half else 1.0
    sur_ref = sur_pol["policy_ref"]

    # ===== 附加税（以实际缴纳增值税为基础）=====
    urban_nom = sur_pol["urban_construction_nominal"]
    edu_nom = sur_pol["education_nominal"]
    local_nom = sur_pol["local_education_nominal"]

    urban_tax = round(vat * urban_nom * half, 2)
    edu_surcharge = round(vat * edu_nom * half, 2)
    local_edu = round(vat * local_nom * half, 2)
    total_surcharge = round(urban_tax + edu_surcharge + local_edu, 2)

    if six_two_half:
        surcharge_policy = (
            f"「六税两费减半」城建{urban_nom*100:.0f}%→{urban_nom*half*100:.1f}%、"
            f"教育{edu_nom*100:.0f}%→{edu_nom*half*100:.1f}%、"
            f"地方教育{local_nom*100:.0f}%→{local_nom*half*100:.0f}%，"
            f"合计{(urban_nom+edu_nom+local_nom)*half*100:.0f}%"
        )
    else:
        surcharge_policy = f"标准附加税率：城建{urban_nom*100:.0f}% + 教育{edu_nom*100:.0f}% + 地方教育{local_nom*100:.0f}%，合计{(urban_nom+edu_nom+local_nom)*100:.0f}%"

    return {
        "季度含税收入": round(revenue, 2),
        "季度不含税收入": revenue_excl,
        "增值税名义税率": vat_rate,
        "增值税实际税率": vat_effective_rate,
        "是否小规模纳税人": "是" if is_small_scale else "否",
        "是否小型微利企业": "是" if is_small_low_profit else "否",
        "是否享受六税两费减半": "是" if six_two_half else "否",
        "增值税应缴": vat,
        "增值税免税说明": vat_note,
        "增值税优惠依据": vat_policy,
        "城建税名义": round(vat * urban_nom, 2),
        "城建税(7%)": urban_tax,
        "教育费附加名义": round(vat * edu_nom, 2),
        "教育费附加(3%)": edu_surcharge,
        "地方教育附加名义": round(vat * local_nom, 2),
        "地方教育附加(2%)": local_edu,
        "附加税合计": total_surcharge,
        "附加税优惠说明": surcharge_policy,
        "六税两费减免金额": round(vat * (urban_nom + edu_nom + local_nom) * (1 - half), 2),
        "增值税及附加合计": round(vat + total_surcharge, 2),
        "_政策区间": pol["_meta"]["period"],
    }


def calc_corporate_income_tax_quarterly(
    revenue: float,
    cost: float,
    period_profit: float,
    ytd_profit: float,
    num_employees: int,
    total_assets: float,
    tax_paid_ytd: float = 0.0,
    vat_data: dict = None,
    tax_year: int = None,
) -> dict:
    """
    计算小型微利企业所得税季度预缴
    返回申报底稿数据字典（匹配企业所得税预缴申报表A类格式）

    税率参数从 tax_policies.json 读取。

    参数：
      revenue:        季度营业收入（元）      申报表第1行
      cost:           季度营业成本（元）      申报表第2行
      period_profit:  季度利润总额（元）      申报表第3行
      ytd_profit:     本年累计利润总额（元）
      num_employees:  季度平均从业人数
      total_assets:   季度平均资产总额（万元）
      tax_paid_ytd:   本年累计已预缴所得税额（元）
      vat_data:       增值税及附加税计算结果
      tax_year:       申报年度（用于匹配政策区间）
    """
    pol = load_tax_policies(tax_year if tax_year else None)
    cit_pol = pol["corporate_income_tax"]
    criteria = cit_pol["small_low_profit_criteria"]
    standard_rate = cit_pol["standard_rate"]
    effective_rate = cit_pol["small_low_profit_effective_rate"]
    cit_ref = cit_pol["policy_ref"]

    # 判断是否符合小型微利企业条件
    is_small_low_profit = (
        num_employees <= criteria["max_employees"]
        and total_assets <= criteria["max_assets_wan"]
    )

    # 实际利润额（申报表第8行）
    actual_profit = period_profit
    period_taxable = max(actual_profit, 0)

    # 第10行：应纳税额 = 应纳税所得额 × 标准税率
    tax_before_relief = round(period_taxable * standard_rate, 2)

    # 第11行：减免所得税额
    if is_small_low_profit and period_taxable > 0:
        tax_actual = round(period_taxable * effective_rate, 2)
        relief = round(tax_before_relief - tax_actual, 2)
    else:
        tax_actual = tax_before_relief
        relief = 0.0

    # 第13行：本期应补（退）所得税额
    tax_payable = round(tax_actual - tax_paid_ytd, 2)
    if tax_payable < 0:
        tax_payable = 0.0

    result = {
        "营业收入": round(revenue, 2),
        "营业成本": round(cost, 2),
        "利润总额": round(period_profit, 2),
        "实际利润额": round(actual_profit, 2),
        "应纳税所得额": period_taxable,
        "标准税率": standard_rate,
        "优惠实际税率": effective_rate,
        "应纳税额_标准": tax_before_relief,
        "减免所得税额": relief,
        "本期应纳税额": tax_actual,
        "本年累计已预缴": tax_paid_ytd,
        "本期应补(退)税额": tax_payable,
        "从业人数": num_employees,
        "资产总额_万元": total_assets,
        "是否小型微利企业": "是" if is_small_low_profit else "否",
        "_政策依据": cit_ref,
        "_政策区间": pol["_meta"]["period"],
    }

    # 附加税汇总（如果有）
    if vat_data:
        result["增值税应缴"] = vat_data.get("增值税应缴", 0.0)
        result["附加税合计"] = vat_data.get("附加税合计", 0.0)
        result["本期税费合计"] = round(
            tax_payable
            + vat_data.get("增值税应缴", 0.0)
            + vat_data.get("附加税合计", 0.0),
            2
        )
    else:
        result["增值税应缴"] = 0.0
        result["附加税合计"] = 0.0
        result["本期税费合计"] = tax_payable

    return result


def get_tax_policy_summary(
    is_small_scale: bool = True,
    is_small_low_profit: bool = True,
    quarter_revenue: float = 0.0,
    num_employees: int = 1,
    total_assets: float = 0.0,
    quarter: int = 1,
) -> dict:
    """
    汇总当前企业适用的全部税收优惠政策

    政策描述优先从 tax_policies.json 读取。
    """
    pol = load_tax_policies()
    vat_pol = pol["vat"]
    sur_pol = pol["surcharges"]
    cit_pol = pol["corporate_income_tax"]
    def_pol = pol["disabled_employment_fund"]
    period_label = pol["_meta"]["period"]
    effective_until = pol["_meta"]["effective_until"]

    policies = []

    # 1. 增值税优惠
    if is_small_scale:
        threshold = vat_pol["quarterly_exempt_threshold"]
        revenue_excl = quarter_revenue / (1 + vat_pol["small_scale_nominal_rate"]) if quarter_revenue else 0
        if quarter_revenue <= 0 or revenue_excl <= threshold:
            policies.append({
                "税种": "增值税",
                "优惠名称": "小规模纳税人季度免税",
                "优惠内容": f"季度不含税收入 ≤ {threshold//10000}万元，免征增值税",
                "政策依据": vat_pol["policy_ref"],
                "适用条件": f"小规模纳税人 + 季收入≤{threshold//10000}万",
                "优惠力度": "100% 免征",
                "减免金额": 0.0,
            })
        else:
            eff = vat_pol["small_scale_effective_rate"]
            nom = vat_pol["small_scale_nominal_rate"]
            policies.append({
                "税种": "增值税",
                "优惠名称": f"小规模纳税人减按{eff*100:.0f}%征收",
                "优惠内容": f"适用{nom*100:.0f}%征收率的应税销售收入，减按{eff*100:.0f}%征收增值税",
                "政策依据": vat_pol["policy_ref"],
                "适用条件": f"小规模纳税人，季收入超{threshold//10000}万",
                "优惠力度": f"{nom*100:.0f}% → {eff*100:.0f}%（有效降低{round((1-eff/nom)*100):.0f}%）",
                "减免金额": 0.0,
            })

    # 2. 六税两费减半
    if sur_pol.get("six_two_half_enabled", True) and (is_small_scale or is_small_low_profit):
        policies.append({
            "税种": "城建税",
            "优惠名称": "「六税两费」减半征收",
            "优惠内容": f"城建税减按实际缴纳增值税的 {sur_pol['urban_construction_effective']*100:.1f}%（原{sur_pol['urban_construction_nominal']*100:.0f}%）",
            "政策依据": sur_pol["policy_ref"],
            "适用条件": "小规模纳税人 或 小型微利企业 ✅",
            "优惠力度": "减免 50%",
            "减免金额": 0.0,
        })
        policies.append({
            "税种": "教育费附加 + 地方教育附加",
            "优惠名称": "「六税两费」减半征收",
            "优惠内容": f"教育费附加 {sur_pol['education_effective']*100:.1f}%（原{sur_pol['education_nominal']*100:.0f}%）+ 地方教育附加 {sur_pol['local_education_effective']*100:.0f}%（原{sur_pol['local_education_nominal']*100:.0f}%）",
            "政策依据": sur_pol["policy_ref"],
            "适用条件": "小规模纳税人 或 小型微利企业 ✅",
            "优惠力度": "减免 50%",
            "减免金额": 0.0,
        })

    # 3. 企业所得税小型微利
    if is_small_low_profit:
        eff = cit_pol["small_low_profit_effective_rate"]
        std = cit_pol["standard_rate"]
        criteria = cit_pol["small_low_profit_criteria"]
        policies.append({
            "税种": "企业所得税",
            "优惠名称": "小型微利企业所得税优惠",
            "优惠内容": f"减按25%计入应纳税所得额，按20%税率缴纳，实际税负 {eff*100:.0f}%",
            "政策依据": cit_pol["policy_ref"],
            "适用条件": f"年利润≤{criteria['max_annual_taxable_income']//10000}万 + 员工≤{criteria['max_employees']}人 + 资产≤{criteria['max_assets_wan']}万 ✅",
            "优惠力度": f"{std*100:.0f}% → {eff*100:.0f}%（有效降低{round((1-eff/std)*100):.0f}%）",
            "减免金额": 0.0,
        })

    # 4. 残保金
    micro_threshold = def_pol["micro_exempt_threshold"]
    if num_employees <= micro_threshold:
        policies.append({
            "税种": "残疾人就业保障金",
            "优惠名称": "小微企业残保金免征",
            "优惠内容": f"在职职工总数 ≤ {micro_threshold}人，免征残疾人就业保障金",
            "政策依据": def_pol["policy_ref"],
            "适用条件": f"员工 {num_employees}人 ≤ {micro_threshold}人 ✅",
            "优惠力度": "100% 免征",
            "减免金额": 0.0,
        })

    return {
        "policies": policies,
        "title": "湖北省/武汉市 税收优惠政策适用清单",
        "valid_until": f"以上政策有效期至 {effective_until}（政策区间：{period_label}）",
        "tip": "以上优惠政策申报时系统自动识别减免，湖北省已实现「免申即享」，无需额外申请备案。",
    }


def calc_disabled_employment_fund(
    prev_year_employees: int,
    prev_year_disabled_employees: int = 0,
    prev_year_avg_salary: float = 0.0,
    local_avg_salary: float = 0.0,
    year: int = None,
) -> dict:
    """
    计算残疾人就业保障金（残保金）

    税率参数从 tax_policies.json 读取。

    参数：
      prev_year_employees:         上年用人单位在职职工人数
      prev_year_disabled_employees: 上年实际安排的残疾人就业人数
      prev_year_avg_salary:         上年用人单位在职职工年平均工资（元）
      local_avg_salary:             当地社会平均工资（元，用于2倍封顶）
      year:                         申报年份
    """
    if year is None:
        year = datetime.now().year

    pol = load_tax_policies(year)
    def_pol = pol["disabled_employment_fund"]
    required_ratio = def_pol["required_ratio"]
    micro_threshold = def_pol["micro_exempt_threshold"]
    cap_mult = def_pol["salary_cap_multiplier"]
    tiers = def_pol["tier_reduction"]
    def_ref = def_pol["policy_ref"]

    # ===== 1. 小微企业免征 =====
    if prev_year_employees <= micro_threshold:
        return {
            "申报年度": year,
            "上年职工人数": prev_year_employees,
            "上年残疾职工人数": prev_year_disabled_employees,
            "上年职工年均工资": round(prev_year_avg_salary, 2),
            "法定安排比例": f"{required_ratio*100:.1f}%",
            "应安排人数": round(prev_year_employees * required_ratio, 2),
            "差额人数": 0,
            "工资计算基数": 0.0,
            "分档征收比例": "免征",
            "应缴残保金（全额）": 0.0,
            "减免金额": 0.0,
            "是否小微企业免征": "是 ✅",
            "免征条件": f"在职职工 {prev_year_employees}人 ≤ {micro_threshold}人",
            "应缴残保金": 0.0,
            "减免金额": 0.0,
            "优惠政策": f"小微企业免征（{def_ref}）",
            "政策依据": def_ref,
            "申报要求": "仍需零申报（进入电子税务局填写后系统自动计算为0）",
            "申报截止": f"通常在 {year} 年 7~9 月（以当地残联公告为准）",
            "计算说明": f"员工 {prev_year_employees}人 ≤ {micro_threshold}人 → 全额免征",
            "_政策区间": pol["_meta"]["period"],
        }

    # ===== 2. 分步计算（>micro_threshold人）=====
    required_disabled = prev_year_employees * required_ratio
    actual_ratio = prev_year_disabled_employees / prev_year_employees if prev_year_employees > 0 else 0

    salary_cap = local_avg_salary * cap_mult if local_avg_salary > 0 else float('inf')
    calc_salary = min(prev_year_avg_salary, salary_cap)

    gap = required_disabled - prev_year_disabled_employees

    # 已达标 → 全额免征
    if gap <= 0:
        payable = 0.0
        exempted = 0.0
        reduction_rate = 1.0
        reduction_note = f"已达标安排比例（实际 {actual_ratio:.2%} ≥ {required_ratio*100:.1f}%），免征残保金"
        base_amount = 0.0
    else:
        base_amount = gap * calc_salary

        # 分档减征
        full = tiers["full_compliance"]
        partial = tiers["partial_compliance"]
        non = tiers["non_compliance"]

        if actual_ratio >= partial["min_ratio"]:
            reduction_rate = partial["rate"]
            reduction_note = partial["label"]
        else:
            reduction_rate = non["rate"]
            reduction_note = non["label"]

        payable = round(base_amount * reduction_rate, 2)
        exempted = round(base_amount - payable, 2)

    return {
        "申报年度": year,
        "上年职工人数": prev_year_employees,
        "上年残疾职工人数": prev_year_disabled_employees,
        "上年职工年均工资": round(prev_year_avg_salary, 2),
        "法定安排比例": f"{required_ratio*100:.1f}%",
        "应安排人数": round(required_disabled, 4),
        "实际安排比例": f"{actual_ratio:.2%}",
        "差额人数": round(gap, 4),
        "工资计算基数": round(calc_salary, 2),
        "工资封顶说明": f"当地社平工资×{cap_mult}={salary_cap:,.2f}元" if local_avg_salary > 0 else "未设置封顶",
        "是否小微企业免征": "否",
        "应缴残保金（全额）": round(base_amount, 2),
        "分档征收比例": f"{reduction_rate:.0%}",
        "分档说明": reduction_note,
        "应缴残保金": payable,
        "减免金额": exempted,
        "优惠政策": reduction_note,
        "政策依据": def_ref,
        "申报要求": f"申报并缴纳 {payable:,.2f} 元",
        "申报截止": f"通常在 {year} 年 7~9 月（以当地残联公告为准）",
        "计算说明": f"({prev_year_employees}人 × {required_ratio*100:.1f}% - {prev_year_disabled_employees}人) × {calc_salary:,.2f}元 × {reduction_rate:.0%} = {payable:,.2f} 元",
        "_政策区间": pol["_meta"]["period"],
    }


def calc_stamp_duty(
    registered_capital: float = 0.0,
    capital_increase: float = 0.0,
    capital_reserve: float = 0.0,
    purchase_amount: float = 0.0,
    loan_amount: float = 0.0,
    tech_amount: float = 0.0,
    property_lease_amount: float = 0.0,
    is_small_low_profit: bool = True,
    tax_year: int = None,
) -> dict:
    """
    计算印花税

    税率参数从 tax_policies.json 读取，按申报年度自动匹配。

    参数：
      registered_capital:  本期实收资本变动（元）
      capital_increase:    本期增资额（元）
      capital_reserve:     资本公积变动（元）
      purchase_amount:     本期购销合同金额（元）
      loan_amount:         本期借款合同金额（元）
      tech_amount:         本期技术合同金额（元）
      property_lease_amount: 本期财产租赁合同金额（元）
      is_small_low_profit: 是否小型微利企业
      tax_year:            申报年度
    """
    pol = load_tax_policies(tax_year if tax_year else None)
    sd_pol = pol["stamp_duty"]
    half_enabled = sd_pol.get("half_enabled", True) and is_small_low_profit
    half = 0.5 if half_enabled else 1.0
    categories = sd_pol["categories"]
    sd_ref = sd_pol["policy_ref"]

    # ===== 各税目计算 =====
    items = []

    def _fmt_rate(nominal, effective):
        """格式化税率显示"""
        return f"{nominal*100:.3f}%（万分之{nominal*10000:.1f}）", f"{effective*100:.4f}%（万分之{effective*10000:.2f}）"

    # 1. 资金账簿
    cap_cat = categories["capital_book"]
    capital_base = registered_capital + capital_increase + capital_reserve
    capital_tax = round(capital_base * cap_cat["effective_rate"], 2)
    if capital_base > 0:
        nr, er = _fmt_rate(cap_cat["nominal_rate"], cap_cat["effective_rate"])
        items.append({
            "税目": cap_cat["name"],
            "品类": cap_cat["basis"],
            "名义税率": nr,
            "优惠后税率": er,
            "计税基础（元）": capital_base,
            "应纳税额（元）": capital_tax,
            "说明": f"注册资本到位/增资 {capital_base:,.2f} 元" + (" 六税两费减半" if half_enabled else ""),
        })

    # 2. 购销合同
    pur_cat = categories["purchase_contract"]
    purchase_tax = round(purchase_amount * pur_cat["effective_rate"], 2)
    if purchase_amount > 0:
        nr, er = _fmt_rate(pur_cat["nominal_rate"], pur_cat["effective_rate"])
        items.append({
            "税目": pur_cat["name"],
            "品类": pur_cat["basis"],
            "名义税率": nr,
            "优惠后税率": er,
            "计税基础（元）": purchase_amount,
            "应纳税额（元）": purchase_tax,
            "说明": f"购销金额 {purchase_amount:,.2f} 元" + (" 六税两费减半" if half_enabled else ""),
        })

    # 3. 借款合同
    loan_cat = categories["loan_contract"]
    loan_tax = round(loan_amount * loan_cat["effective_rate"], 2)
    if loan_amount > 0:
        nr, er = _fmt_rate(loan_cat["nominal_rate"], loan_cat["effective_rate"])
        items.append({
            "税目": loan_cat["name"],
            "品类": loan_cat["basis"],
            "名义税率": nr,
            "优惠后税率": er,
            "计税基础（元）": loan_amount,
            "应纳税额（元）": loan_tax,
            "说明": f"借款金额 {loan_amount:,.2f} 元" + (" 六税两费减半" if half_enabled else ""),
        })

    # 4. 技术合同
    tech_cat = categories["tech_contract"]
    tech_tax = round(tech_amount * tech_cat["effective_rate"], 2)
    if tech_amount > 0:
        nr, er = _fmt_rate(tech_cat["nominal_rate"], tech_cat["effective_rate"])
        items.append({
            "税目": tech_cat["name"],
            "品类": tech_cat["basis"],
            "名义税率": nr,
            "优惠后税率": er,
            "计税基础（元）": tech_amount,
            "应纳税额（元）": tech_tax,
            "说明": f"技术合同金额 {tech_amount:,.2f} 元" + (" 六税两费减半" if half_enabled else ""),
        })

    # 5. 财产租赁合同
    prop_cat = categories["property_lease"]
    property_tax = round(property_lease_amount * prop_cat["effective_rate"], 2)
    if property_lease_amount > 0:
        nr, er = _fmt_rate(prop_cat["nominal_rate"], prop_cat["effective_rate"])
        items.append({
            "税目": prop_cat["name"],
            "品类": prop_cat["basis"],
            "名义税率": nr,
            "优惠后税率": er,
            "计税基础（元）": property_lease_amount,
            "应纳税额（元）": property_tax,
            "说明": f"租赁金额 {property_lease_amount:,.2f} 元" + (" 六税两费减半" if half_enabled else ""),
        })

    total_stamp_duty = round(sum(i["应纳税额（元）"] for i in items), 2)
    nominal_total = round(sum(i["计税基础（元）"] * cats_nominal_rate(categories, i["税目"]) for i in items), 2)
    relief = round(nominal_total - total_stamp_duty, 2)

    return {
        "明细": items,
        "税目数量": len(items),
        "印花税合计（名义）": nominal_total,
        "六税两费减免": relief,
        "印花税合计（应缴）": total_stamp_duty,
        "是否六税两费减半": "是" if half_enabled else "否",
        "政策依据": sd_ref,
        "申报方式": "按次或按期汇总 → 湖北省电子税务局 → 「印花税申报」",
        "提示": "资金账簿仅在初始到位或增资时缴纳，已缴部分不重复征收",
        "_政策区间": pol["_meta"]["period"],
    }


def cats_nominal_rate(categories: dict, name: str) -> float:
    """从 categories 反向查找名义税率（用于减免金额计算）"""
    for key, cat in categories.items():
        if cat["name"] == name:
            return cat["nominal_rate"]
    return 0.0


def format_corporate_tax_report(result: dict, quarter: int, year: int, vat_data: dict = None) -> str:
    """生成企业所得税及税费测算申报说明文字（匹配A200000格式）"""
    lines = [
        f"{'='*70}",
        f"  {year}年第{quarter}季度 企业所得税预缴 + 增值税及附加测算说明",
        f"  （匹配 A200000 申报表格式 | 含增值税/城建税/教育费附加）",
        f"{'='*70}",
        "",
        f"一、基本信息",
        f"  纳税人名称：    湖北启贤托儿所有限公司",
        f"  所属期间：      {year}年{[1,4,7,10][quarter-1]}月01日 至 {year}年{[3,6,9,12][quarter-1]}月31日",
        f"  企业类型：     {result['是否小型微利企业']}（小型微利企业）",
        f"  从业人数：     {result['从业人数']} 人",
        f"  资产总额：     {result['资产总额_万元']:.2f} 万元",
        "",
        f"{'─'*50}",
        f"二、收入成本利润（申报表第1~3行）",
        f"{'─'*50}",
        f"  第1行 营业收入：       {result['营业收入']:>15,.2f} 元",
        f"  第2行 营业成本：       {result['营业成本']:>15,.2f} 元",
        f"  第3行 利润总额：       {result['利润总额']:>15,.2f} 元",
        "",
        f"{'─'*50}",
        f"三、应纳税所得额计算（申报表第4~8行）",
        f"{'─'*50}",
        f"  第4行 特定业务调整：    {0:>15,.2f}",
        f"  第5行 不征税收入：       {0:>15,.2f}",
        f"  第6行 固定资产折旧调整： {0:>15,.2f}",
        f"  第7行 弥补以前年度亏损： {0:>15,.2f}",
        f"  ───────────────────────────────",
        f"  第8行 实际利润额：       {result['实际利润额']:>15,.2f} 元",
        "",
        f"{'─'*50}",
        f"四、税款计算（申报表第9~13行）",
        f"{'─'*50}",
        f"  第9行 税率（25%）：       {'25%':>15s}",
        f"  第10行 应纳所得税额：     {result['应纳税额_标准']:>15,.2f} 元",
        f"  第11行 减免所得税额：     {result['减免所得税额']:>15,.2f} 元",
        f"  第12行 本年累计已预缴：   {result['本年累计已预缴']:>15,.2f} 元",
        f"  ───────────────────────────────",
        f"  第13行 本期应补(退)税额： {result['本期应补(退)税额']:>15,.2f} 元",
        "",
        f"{'─'*50}",
        f"五、计算说明",
        f"{'─'*50}",
    ]

    if result['利润总额'] <= 0:
        lines.extend([
            f"  【本期亏损】利润总额为 {result['利润总额']:,.2f} 元（负数），",
            f"             实际利润额取0或保留负数，无需缴纳企业所得税。",
            f"",
            f"  第10行应纳所得税额 = max(实际利润额, 0) × 25% = 0 元",
            f"  第11行减免所得税额 = 0 元（亏损无减免）",
            f"  第13行本期应补退税额 = 0 元",
        ])
    else:
        if result['是否小型微利企业'] == '是':
            lines.extend([
                f"  【小型微利企业优惠】2024-2027年政策：",
                f"  - 减按25%计入应纳税所得额，按20%税率征收",
                f"  - 实际税负 = 25% × 20% = 5%",
                f"",
                f"  第10行应纳所得税额 = {result['应纳税所得额']:,.2f} × 25% = {result['应纳税额_标准']:,.2f} 元",
                f"  第11行减免所得税额 = {result['应纳税额_标准']:,.2f} - {result['本期应纳税额']:,.2f} = {result['减免所得税额']:,.2f} 元",
                f"  第13行本期应补退税额 = {result['本期应纳税额']:,.2f} - {result['本年累计已预缴']:,.2f} = {result['本期应补(退)税额']:,.2f} 元",
            ])
        else:
            lines.extend([
                f"  【一般企业】适用标准税率 25%",
                f"  第10行应纳所得税额 = {result['应纳税所得额']:,.2f} × 25% = {result['应纳税额_标准']:,.2f} 元",
                f"  第11行减免所得税额 = 0 元",
                f"  第13行本期应补退税额 = {result['本期应纳税额']:,.2f} 元",
            ])

    lines.extend([
        "",
        f"{'─'*50}",
        f"六、增值税及附加税测算（参考）",
        f"{'─'*50}",
    ])

    if vat_data:
        lines.extend([
            f"  增值税类型：    {'小规模纳税人3%' if vat_data['是否小规模纳税人']=='是' else '一般纳税人'}",
            f"  季度含税收入：  {vat_data['季度含税收入']:>15,.2f} 元",
            f"  季度不含税收入：{vat_data['季度不含税收入']:>15,.2f} 元",
            f"  增值税说明：    {vat_data['增值税免税说明']}",
            f"  增值税应缴：    {vat_data['增值税应缴']:>15,.2f} 元",
            f"  ───────────────────────────────",
            f"  城建税(7%)：    {vat_data['城建税(7%)']:>15,.2f} 元",
            f"  教育费附加(3%)：{vat_data['教育费附加(3%)']:>15,.2f} 元",
            f"  地方教育附加(2%)：{vat_data['地方教育附加(2%)']:>13,.2f} 元",
            f"  附加税合计：    {vat_data['附加税合计']:>15,.2f} 元",
            f"  ───────────────────────────────",
            f"  增值税+附加合计：{vat_data['增值税及附加合计']:>14,.2f} 元",
        ])
    else:
        lines.append("  （未录入增值税信息，请在申报界面填写季度收入后测算）")

    lines.extend([
        "",
        f"{'─'*50}",
        f"七、本期税费汇总",
        f"{'─'*50}",
        f"  企业所得税（本期应补缴）：{result['本期应补(退)税额']:>12,.2f} 元",
        f"  增值税应缴：              {result.get('增值税应缴', 0.0):>12,.2f} 元",
        f"  附加税合计：              {result.get('附加税合计', 0.0):>12,.2f} 元",
        f"  ───────────────────────────────",
        f"  本期税费合计：            {result.get('本期税费合计', result['本期应补(退)税额']):>12,.2f} 元",
        "",
        f"{'─'*50}",
        f"八、申报提醒",
        f"{'─'*50}",
        "  1. 请核对利润总额与利润表（小企业会计准则）一致；",
        "  2. 小型微利企业优惠由系统自动判别，无需额外备案；",
        "  3. 申报截止时间为季度终了后15日内（4月、7月、10月、次年1月15日前）；",
        "  4. 请及时在国家税务总局湖北省电子税务局完成预缴申报。",
        "",
        f"{'='*70}",
        f"  —— 由 启贤托育AI税务助手 自动生成 · {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"{'='*70}",
    ])
    return "\n".join(lines)


# ===============================================
#  银行流水自动分类（遵循小企业会计准则）
# ===============================================

def classify_bank_transaction(desc: str) -> dict:
    """
    根据银行流水摘要，自动分类到小企业会计准则利润表或资产负债表项目
    返回：{"category": "营业收入", "pl_item": "营业收入", "type": "收入", "account": "主营业务收入"}
    """
    desc = str(desc).lower()
    
    # ═════════════════════════════════════════════
    #  资产负债表项目（不进入利润表）
    # ═════════════════════════════════════════════
    
    # 其他应付款 — 股东借款 / 股东垫款（负债）
    shareholder_loan_kw = ["股东借款", "股东垫款", "股东暂借", "股东往来款",
                           "老板借款", "老板垫款", "老板暂借", "老板往来",
                           "法人借款", "法人垫款", "法人往来", "法人暂借",
                           "还款给股东", "还股东款", "归还股东", "还老板款",
                           "股东转入", "老板转入", "法人转入"]
    if any(k in desc for k in shareholder_loan_kw):
        return {
            "category": "其他应付款-股东借款",
            "pl_item": "资产负债表-其他应付款",
            "type": "负债",
            "account": "其他应付款-股东借款"
        }
    
    # 实收资本 — 股东注资（所有者权益）
    if any(k in desc for k in ["注册资本", "实收资本", "增资款", "投资款注入",
                                "股东出资", "股本注入"]):
        return {
            "category": "实收资本",
            "pl_item": "资产负债表-实收资本",
            "type": "所有者权益",
            "account": "实收资本"
        }
    
    # ═════════════════════════════════════════════
    #  利润表项目
    # ═════════════════════════════════════════════
    
    # 一、营业收入（利润表第1行）
    if any(k in desc for k in ["货款", "销售收入", "服务费", "咨询费", "收款", 
                               "主营业务收入", "其他业务收入", "销售", "服务收入"]):
        return {
            "category": "营业收入",
            "pl_item": "营业收入",
            "type": "收入",
            "account": "主营业务收入"
        }
    
    # 二、营业成本（利润表第2行）
    elif any(k in desc for k in ["采购", "进货", "材料成本", "主营业务成本", 
                                 "材料", "成本", "进货成本"]):
        return {
            "category": "营业成本",
            "pl_item": "营业成本",
            "type": "支出",
            "account": "主营业务成本"
        }
    
    # 三、税金及附加（利润表第3行）
    elif any(k in desc for k in ["税金", "附加费", "城建税", "教育费附加", 
                                 "地方教育附加", "印花税", "房产税", "土地使用税"]):
        return {
            "category": "税金及附加",
            "pl_item": "税金及附加",
            "type": "支出",
            "account": "税金及附加"
        }
    
    # 四、销售费用（利润表第11行）
    elif any(k in desc for k in ["广告", "推广", "营销", "宣传", "展览", 
                                 "运输费", "装卸费", "包装费", "销售佣金", "促销"]):
        return {
            "category": "销售费用",
            "pl_item": "销售费用",
            "type": "支出",
            "account": "销售费用"
        }
    
    # 五、管理费用（利润表第14行）
    elif any(k in desc for k in ["工资", "社保", "公积金", "福利费", "奖金", 
                                 "养老金", "医保", "医疗保险", "生育保险", "失业保险", "工伤保险"]):
        return {
            "category": "管理费用-人工",
            "pl_item": "管理费用",
            "type": "支出",
            "account": "管理费用-工资社保"
        }
    
    elif any(k in desc for k in ["水电", "物业", "房租", "租赁", "办公用品", 
                                 "电话费", "网络费", "维修费"]):
        return {
            "category": "管理费用-办公",
            "pl_item": "管理费用",
            "type": "支出",
            "account": "管理费用-办公费"
        }
    
    elif any(k in desc for k in ["差旅", "交通", "餐饮", "招待", "会议费", 
                                 "培训费", "咨询费", "审计费"]):
        return {
            "category": "管理费用-其他",
            "pl_item": "管理费用",
            "type": "支出",
            "account": "管理费用-业务招待费"
        }
    
    # 六、财务费用（利润表第18行）
    elif any(k in desc for k in ["利息", "手续费", "银行手续费", "汇款手续费", 
                                 "贷款利息", "存款利息", "结息"]):
        return {
            "category": "财务费用",
            "pl_item": "财务费用",
            "type": "支出",
            "account": "财务费用-手续费"
        }
    
    # 七、投资收益（利润表第20行）
    elif any(k in desc for k in ["分红", "投资收益", "股息", "理财收益", 
                                 "投资收入"]):
        return {
            "category": "投资收益",
            "pl_item": "投资收益",
            "type": "收入",
            "account": "投资收益"
        }
    
    # 八、营业外收入（利润表第22行）
    elif any(k in desc for k in ["政府补助", "补贴", "罚款收入", "违约金收入", 
                                 "捐赠收入", "盘盈",
                                 "实名认证", "认证打款", "账户验证", "打款验证",
                                 "验证款", "认证验证"]):
        return {
            "category": "营业外收入",
            "pl_item": "营业外收入",
            "type": "收入",
            "account": "营业外收入"
        }
    
    # 九、营业外支出（利润表第24行）
    elif any(k in desc for k in ["罚款", "捐赠", "损失", "盘亏", "自然灾害损失", 
                                 "违约金支出"]):
        return {
            "category": "投资收益",
            "pl_item": "投资收益",
            "type": "收入",
            "account": "投资收益"
        }
    
    else:
        return {
            "category": "待分类",
            "pl_item": "待分类",
            "type": "未知",
            "account": "待确认"
        }


def generate_profit_statement(df_txns: object) -> dict:
    """
    根据银行流水DataFrame，生成小企业会计准则利润表（会小企02表）
    df_txns 必须包含列：["摘要", "收入金额", "支出金额", "自动分类"]
    
    返回利润表各项目金额（单位：元），行次对应 会小企02表。
    官方行次：1营业收入, 2营业成本, 3税金及附加, 11销售费用,
              14管理费用, 18财务费用, 20投资收益, 21营业利润,
              22营业外收入, 24营业外支出, 30利润总额, 31所得税费用, 32净利润
    """
    # 第1行：营业收入
    revenue = df_txns[df_txns["自动分类"] == "营业收入"]["收入金额"].sum()
    
    # 第2行：营业成本
    cost = df_txns[df_txns["自动分类"] == "营业成本"]["支出金额"].sum()
    
    # 第3行：税金及附加
    tax_expense = df_txns[df_txns["自动分类"] == "税金及附加"]["支出金额"].sum()
    
    # 第11行：销售费用
    selling_expense = df_txns[
        df_txns["自动分类"].str.contains("销售费用", na=False)
    ]["支出金额"].sum()
    
    # 第14行：管理费用 = 所有管理费用子分类的支出金额之和
    manage_expense = df_txns[
        df_txns["自动分类"].str.contains("管理费用", na=False)
    ]["支出金额"].sum()
    
    # 第18行：财务费用（含利息收入，以负数列示）
    finance_expense = df_txns[df_txns["自动分类"] == "财务费用"]["支出金额"].sum()
    finance_income = df_txns[df_txns["自动分类"] == "财务费用"]["收入金额"].sum()
    finance_net = round(finance_expense - finance_income, 2)  # 利息收入抵减
    
    # 第20行：投资收益
    investment_income = df_txns[df_txns["自动分类"] == "投资收益"]["收入金额"].sum()
    
    # 第21行：营业利润 = 1 - 2 - 3 - 11 - 14 - 18 + 20
    operating_profit = round(
        revenue - cost - tax_expense - selling_expense
        - manage_expense - finance_net + investment_income, 2
    )
    
    # 第22行：营业外收入
    other_income = df_txns[df_txns["自动分类"] == "营业外收入"]["收入金额"].sum()
    
    # 第24行：营业外支出
    other_expense = df_txns[df_txns["自动分类"] == "营业外支出"]["支出金额"].sum()
    
    # 第30行：利润总额 = 21 + 22 - 24
    total_profit = round(operating_profit + other_income - other_expense, 2)
    
    # 第31行：所得税费用（亏损为0）
    if total_profit > 0:
        pol = load_tax_policies()
        eff_rate = pol["corporate_income_tax"]["small_low_profit_effective_rate"]
        income_tax = round(total_profit * eff_rate, 2)
    else:
        income_tax = 0.0
    
    # 第32行：净利润 = 30 - 31
    net_profit = round(total_profit - income_tax, 2)
    
    # ═══ 资产负债表项目（不影响利润）═══
    # 其他应付款-股东借款：收入=借款增加，支出=还款减少
    s_loan_in = df_txns[df_txns["自动分类"] == "其他应付款-股东借款"]["收入金额"].sum()
    s_loan_out = df_txns[df_txns["自动分类"] == "其他应付款-股东借款"]["支出金额"].sum()
    s_loan_net = round(s_loan_in - s_loan_out, 2)  # 净借款增加
    
    # 实收资本
    capital_in = df_txns[df_txns["自动分类"] == "实收资本"]["收入金额"].sum()
    capital_out = df_txns[df_txns["自动分类"] == "实收资本"]["支出金额"].sum()
    capital_net = round(capital_in - capital_out, 2)
    
    # 待分类
    unclassified_in = df_txns[df_txns["自动分类"] == "待分类"]["收入金额"].sum()
    unclassified_out = df_txns[df_txns["自动分类"] == "待分类"]["支出金额"].sum()
    
    return {
        "营业收入": round(revenue, 2),
        "营业成本": round(cost, 2),
        "税金及附加": round(tax_expense, 2),
        "销售费用": round(selling_expense, 2),
        "管理费用": round(manage_expense, 2),
        "财务费用": finance_net,
        "投资收益": round(investment_income, 2),
        "营业利润": operating_profit,
        "营业外收入": round(other_income, 2),
        "营业外支出": round(other_expense, 2),
        "利润总额": total_profit,
        "所得税费用": income_tax,
        "净利润": net_profit,
        # 资产负债表项目
        "其他应付款_股东借款_收入": round(s_loan_in, 2),
        "其他应付款_股东借款_支出": round(s_loan_out, 2),
        "其他应付款_股东借款_净额": s_loan_net,
        "实收资本_收入": round(capital_in, 2),
        "实收资本_支出": round(capital_out, 2),
        "实收资本_净额": capital_net,
        "待分类_收入": round(unclassified_in, 2),
        "待分类_支出": round(unclassified_out, 2),
    }


def validate_quarterly_declaration(profit_data: dict, revenue: float, cost: float, period_profit: float) -> list:
    """
    校验利润表数据与企业所得税季度申报表数据是否一致
    
    参数：
      profit_data: generate_profit_statement() 的返回值
      revenue: 申报表第1行 营业收入
      cost: 申报表第2行 营业成本
      period_profit: 申报表第3行 利润总额
    
    返回：校验结果列表，每个元素为 (是否通过, 提示信息)
    """
    results = []
    
    # 校验1：营业收入
    if abs(profit_data["营业收入"] - revenue) > 1:
        results.append((False, f"营业收入不一致：利润表{profit_data['营业收入']:.2f} vs 申报表{revenue:.2f}"))
    else:
        results.append((True, f"营业收入校验通过：{revenue:.2f} 元"))
    
    # 校验2：营业成本
    if abs(profit_data["营业成本"] - cost) > 1:
        results.append((False, f"营业成本不一致：利润表{profit_data['营业成本']:.2f} vs 申报表{cost:.2f}"))
    else:
        results.append((True, f"营业成本校验通过：{cost:.2f} 元"))
    
    # 校验3：利润总额
    if abs(profit_data["利润总额"] - period_profit) > 1:
        results.append((False, f"利润总额不一致：利润表{profit_data['利润总额']:.2f} vs 申报表{period_profit:.2f}"))
    else:
        results.append((True, f"利润总额校验通过：{period_profit:.2f} 元"))
    
    return results


# ===============================================
#  主程序（示例）
# ===============================================
if __name__ == "__main__":
    print("湖北启贤托儿所有限公司（启贤托育）- 个税社保计算工具 v1.0")
    print("（无需安装任何依赖，直接运行）\n")

    employees = [
        {
            "name": "员工A",
            "gross_salary": 10522,
            "si_base": 5000,
            "si_personal_actual": 522,
            "special_deductions": 5000,
            "child_education": 2000,
            "infant_care": 2000,
            "elderly_care": 1000,
        },
        # 增加员工只需复制上方字典，修改姓名和工资金额
        # {
        #     "name": "员工B",
        #     "gross_salary": 8000,
        #     ...
        # },
    ]

    results = process_employees(employees)
    print_results(results)
    export_csv(results)

    print("\n[提示] 使用提示：")
    print("  1. 修改上方 employees 列表，增加/修改员工数据")
    print("  2. 专项附加扣除如有变化，修改 special_deductions 字段")
    print("  3. 运行：python tax_calculator.py")
    print("  4. CSV底稿可直接导入Excel或发送给财务")


# ===============================================
#  工资数据校验（银行流水 / 个税申报 / 年报三部分）
# ===============================================

def validate_salary_data(
    employees: list[dict],
    bank_df: "pd.DataFrame | None" = None,
    tax_filing_df: "pd.DataFrame | None" = None,
    annual_total_salary: float = 0.0,
) -> dict:
    """
    三重校验工资数据，返回校验结果字典。

    参数：
    - employees: 系统录入的员工列表（calc_one_employee 输入格式）
    - bank_df: 银行流水 DataFrame，需含「摘要」「支出金额」列
    - tax_filing_df: 个税申报记录 DataFrame，需含「姓名」「累计收入」列
    - annual_total_salary: 年报中的「全年工资总额」（用于第三重校验）

    返回：
    {
        "bank_match": [...],    # 银行流水 vs 系统工资
        "tax_match": [...],    # 个税申报 vs 系统工资
        "annual_match": {...},  # 年报工资总额 vs 系统年工资合计
        "warnings": [...],     # 所有警告信息
    }
    """
    import pandas as pd

    result = {
        "bank_match": [],
        "tax_match": [],
        "annual_match": {},
        "warnings": [],
    }

    # ── 系统工资合计（年） ──
    sys_annual_total = 0.0
    for emp in employees:
        m = emp.get("gross_salary", 0) or 0
        sys_annual_total += m * 12

    # ============================================================
    #  校验1：银行流水 vs 系统工资
    # ============================================================
    if bank_df is not None and len(bank_df) > 0:
        df = bank_df.copy()

        # 兼容列名
        col_map = {}
        for col in df.columns:
            cl = str(col).strip().lower()
            if any(k in cl for k in ["摘要", "备注", "用途", "description"]):
                col_map[col] = "摘要"
            elif any(k in cl for k in ["支出", "借方", "取款", "debit", "转出"]):
                col_map[col] = "支出金额"
            elif any(k in cl for k in ["金额", "发生额", "transaction"]):
                col_map[col] = "金额"
        df = df.rename(columns=col_map)

        # 用 classify_bank_transaction 识别工资类支出
        def is_salary_row(desc):
            category = classify_bank_transaction(str(desc))["category"]
            return category in ("管理费用-人工",)

        # 工资关键词二次兜底
        salary_keywords = ["工资", "薪资", "绩效", "奖金", "薪酬", "薪水", " salary", "salary", "payroll"]

        def is_salary_desc(desc):
            d = str(desc).lower()
            return any(k in d for k in salary_keywords)

        # 提取支出金额列
        amount_col = None
        for c in ["支出金额", "金额"]:
            if c in df.columns:
                amount_col = c
                break
        if amount_col is None:
            for c in df.columns:
                if any(k in str(c).lower() for k in ["支出", "借方", "amount", "金额"]):
                    amount_col = c
                    break

        if amount_col:
            df["_is_salary"] = df["摘要"].apply(lambda x: is_salary_row(x) or is_salary_desc(x))
            salary_txns = df[df["_is_salary"] == True].copy()

            if len(salary_txns) > 0:
                bank_salary_total = salary_txns[amount_col].astype(float).sum()
                diff = sys_annual_total - bank_salary_total
                pct = (diff / bank_salary_total * 100) if bank_salary_total > 0 else 0

                result["bank_match"] = {
                    "bank_salary_total": round(bank_salary_total, 2),
                    "sys_annual_total": round(sys_annual_total, 2),
                    "diff": round(diff, 2),
                    "diff_pct": round(pct, 2),
                    "match": abs(diff) < max(bank_salary_total * 0.05, 500),  # 5% 或 500 元以内认为一致
                    "txn_count": len(salary_txns),
                }

                if not result["bank_match"]["match"]:
                    result["warnings"].append(
                        f"⚠️ 银行流水工资支出 {bank_salary_total:,.0f} 元 "
                        f"与系统年工资 {sys_annual_total:,.0f} 元不一致（差 {diff:+,.0f} 元，{pct:+.1f}%）"
                    )
                else:
                    result["warnings"].append(
                        f"✅ 银行流水工资支出与系统工资一致（差 {diff:+,.0f} 元）"
                    )
            else:
                result["warnings"].append("⚠️ 银行流水中未识别到工资/奖金类支出，请检查摘要关键词")
        else:
            result["warnings"].append("⚠️ 银行流水文件中未找到支出金额列，无法校验工资")

    # ============================================================
    #  校验2：个税申报记录 vs 系统工资
    # ============================================================
    if tax_filing_df is not None and len(tax_filing_df) > 0:
        df_tax = tax_filing_df.copy()

        # 兼容列名
        col_map2 = {}
        for col in df_tax.columns:
            cl = str(col).strip().lower()
            if "姓名" in col or "name" in cl:
                col_map2[col] = "姓名"
            if any(k in cl for k in ["累计收入", "收入额", "工资薪金", "应纳税所得额", "收入"]):
                col_map2[col] = "累计收入"
        df_tax = df_tax.rename(columns=col_map2)

        if "姓名" in df_tax.columns and "累计收入" in df_tax.columns:
            # 按员工匹配
            emp_map = {e.get("name", ""): e for e in employees}
            for _, row in df_tax.iterrows():
                name = str(row.get("姓名", "")).strip()
                try:
                    tax_income = float(row.get("累计收入", 0) or 0)
                except (ValueError, TypeError):
                    continue
                if name in emp_map:
                    sys_annual = emp_map[name].get("gross_salary", 0) * 12
                    diff = sys_annual - tax_income
                    result["tax_match"].append({
                        "name": name,
                        "sys_annual": round(sys_annual, 2),
                        "tax_filing_income": round(tax_income, 2),
                        "diff": round(diff, 2),
                        "match": abs(diff) < max(sys_annual * 0.01, 100),  # 1% 或 100 元以内
                    })
                    if abs(diff) >= max(sys_annual * 0.01, 100):
                        result["warnings"].append(
                            f"⚠️ 员工「{name}」个税申报累计收入 {tax_income:,.0f} 元 "
                            f"与系统年工资 {sys_annual:,.0f} 元差 {diff:+,.0f} 元"
                        )

            if len(result["tax_match"]) == 0:
                result["warnings"].append("⚠️ 个税申报记录中未找到匹配的员工姓名")
        else:
            result["warnings"].append("⚠️ 个税申报文件中未找到「姓名」和「累计收入」列")

    # ============================================================
    #  校验3：年报工资总额 vs 系统年工资合计
    # ============================================================
    if annual_total_salary and annual_total_salary > 0:
        diff = sys_annual_total - annual_total_salary
        pct = (diff / annual_total_salary * 100) if annual_total_salary > 0 else 0
        result["annual_match"] = {
            "annual_total_salary": round(annual_total_salary, 2),
            "sys_annual_total": round(sys_annual_total, 2),
            "diff": round(diff, 2),
            "diff_pct": round(pct, 2),
            "match": abs(diff) < max(annual_total_salary * 0.03, 1000),  # 3% 或 1000 元以内
        }
        if not result["annual_match"]["match"]:
            result["warnings"].append(
                f"⚠️ 年报工资总额 {annual_total_salary:,.0f} 元 "
                f"与系统年工资合计 {sys_annual_total:,.0f} 元不一致（差 {diff:+,.0f} 元，{pct:+.1f}%）"
            )
        else:
            result["warnings"].append(
                f"✅ 年报工资总额与系统工资一致（差 {diff:+,.0f} 元）"
            )

    return result
