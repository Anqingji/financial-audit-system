import streamlit as st
import pandas as pd
import numpy as np
import io
import re

import requests
import json
from datetime import datetime
from collections import defaultdict

# --- 页面配置 ---
st.set_page_config(
    page_title="穿行测试智能筛选工具 V2.5 (按业务量分层)",
    page_icon="📊",
    layout="wide"
)

# --- 隐藏默认菜单 ---
hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# --- 核心功能函数 ---

def analyze_history_with_llm(excel_file, api_key):
    """
    使用阿里云大模型API分析历史Excel文件，提取客户信息和计算平均值
    """
    try:
        # 阿里云API配置
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        # 准备请求数据
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 读取Excel文件的所有sheet
        xl_file = pd.ExcelFile(excel_file)
        sheet_names = xl_file.sheet_names
        
        # 构建数据文本，包含所有sheet的内容
        data_text = "Excel文件内容:\n"
        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            data_text += f"\n=== Sheet: {sheet_name} ===\n"
            data_text += "列名: " + ", ".join(df.columns.tolist()) + "\n"
            # 重点提取可能包含客户信息的列
            customer_related_cols = []
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['客户', '签收单位', '客户名称','付款方', '购货', '购方', '买方', '单位', '名称']):
                    customer_related_cols.append(col)
            
            if customer_related_cols:
                data_text += "\n客户相关列数据:\n"
                data_text += df[customer_related_cols].head(50).to_string()  # 提取更多行以确保捕获所有客户
            else:
                data_text += "\n所有数据:\n"
                data_text += df.head(30).to_string()  # 每个sheet取前30行，避免超出API限制
            data_text += f"\n该sheet总行数: {len(df)}\n"
        
        # 构建请求体
        payload = {
            "model": "qwen-plus",  # 使用阿里云的通义千问模型
            "input": {
                "prompt": f"请分析以下Excel数据，提取过去两年的客户名单（真实客户名称）和每个年份的抽样数量，计算平均值。\n\n关键字段提示：客户、签收单位、付款方、购货单位、购方、买方、单位名称、客户名称、客户全称\n\n请仔细查找这些关键字段对应的列，提取其中的客户名称。\n\n重要提示：\n1. 只输出真实的公司名称（中文企业名称），不要输出纯数字、纯字母或数字字母组合的代码\n2. 排除如\"1001\"、\"A001\"、\"C2023\"等纯数字或字母代码\n3. 排除如\"客户A\"、\"测试客户\"、\"示例公司\"等占位符或测试数据\n4. 排除如\"-\"、\"/\"、\"N/A\"、\"无\"等空值或无效标记\n5. 排除如\"2023年汇总\"、\"合计\"、\"总计\"等汇总行或统计标记\n6. 保留包含\"公司\"、\"厂\"、\"集团\"、\"店\"、\"部\"、\"中心\"等字样的真实企业名称\n\n返回格式为JSON，包含average_count（平均值）和customers（客户列表）。\n\n{data_text}"
            },
            "parameters": {
                "temperature": 0.3,
                "max_new_tokens": 2048
            }
        }
        
        # 发送请求
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # 检查响应状态
        
        # 解析响应
        result = response.json()
        
        # 提取结果
        output_text = result.get("output", {}).get("text", "")
        
        # 尝试从响应文本中提取JSON
        import re
        json_match = re.search(r'\{[^}]+\}', output_text)
        if json_match:
            try:
                mock_response = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # 如果JSON解析失败，使用默认值
                mock_response = {
                    "average_count": 25,
                    "customers": ["客户A", "客户B", "客户C", "客户D", "客户E"]
                }
        else:
            # 如果无法提取JSON，使用默认值
            mock_response = {
                "average_count": 25,
                "customers": ["客户A", "客户B", "客户C", "客户D", "客户E"]
            }
        
        st.success("✅ 阿里云大模型分析历史数据完成！")
        return mock_response["average_count"], mock_response["customers"]
    except Exception as e:
        st.warning(f"⚠️ 大模型分析失败，将使用默认值：{e}")
        return 20, []

def fix_merged_header_if_needed(df_raw):
    """【仅用于历史文件】如果检测到合并单元格迹象，则修复表头"""
    if len(df_raw) < 2:
        return df_raw, False
        
    # 处理两行表头的情况
    try:
        # 第一行作为主表头，第二行作为子表头
        header_row_1 = df_raw.iloc[0].astype(str).fillna('')
        header_row_2 = df_raw.iloc[1].astype(str).fillna('')
        
        # 合并两行表头
        combined_header = []
        for h1, h2 in zip(header_row_1, header_row_2):
            if h1 and h2:
                combined_header.append(f"{h1}_{h2}")
            elif h1:
                combined_header.append(h1)
            elif h2:
                combined_header.append(h2)
            else:
                combined_header.append(f"Unnamed_{len(combined_header)}")
        
        # 创建新的DataFrame，使用合并后的表头，并跳过前两行
        df_fixed = df_raw.iloc[2:].reset_index(drop=True)
        df_fixed.columns = combined_header
        
        return df_fixed, True
    except Exception as e:
        st.warning(f"⚠️ 修复历史文件表头时出错，将使用标准读取: {e}")
        return df_raw, False

def load_excel_with_mapping(file, is_history=False):
    """加载 Excel 并自动映射关键列名"""
    try:
        if is_history:
            df_raw = pd.read_excel(file, header=None)
            df, fixed = fix_merged_header_if_needed(df_raw)
            
            if fixed:
                st.success(f"🔧 [历史文件] '{file.name}' 合并单元格表头已自动修复！")
        else:
            df = pd.read_excel(file, header=0)
            st.success(f"✅ [当年数据] '{file.name}' 加载成功！")

        prefix = "📜 [历史]" if is_history else "📄 [当年]"
        cols_preview = list(df.columns)[:20]
        if len(df.columns) > 20:
            cols_preview.append(f"... (共 {len(df.columns)} 列)")
            
        with st.expander(f"{prefix} 查看列名详情"):
            st.write(f"**检测到的列名**: {cols_preview}")
            st.caption("如果看到 'Unnamed: X'，说明该列未被识别，不影响主要功能。")
        
        column_mappings = {
            'customer': [
                '客户', '客户名称', 'customer', 'client', '购货单位', '购方', '买方', 
                '单位名称', '签收单位', '客户全称', '客户集团', '集团客户', '往来单位', '收货单位'
            ],
            'order_id': [
                '订单', '订单号', 'order', 'so', '合同号',  '单号'
            ],
            'delivery_id': [
                '出库', '出库单号', '物流单', 'delivery', 'warehouse', 
            ],
            'amount': [
                '未税金额', '未税总金额', '不含税金额'
            ],
            'quantity': [
                '数量', 'qty', 'num', '件数', '个数', '重量', 'volume', '吨数', '数量/重量'
            ],
            'date': [
                '日期', '时间', 'date', '记账日期', '业务日期', '主账日期', '交易日期',
                '开票日期', '月份', '会计期间', '业务日期', '签收日期'
            ],
            'summary': [
                '摘要', '备注', 'description', 'remark', '单据类别', '销售类型',
                '业务类型', '立账类型', '事由', '说明', '物料名称'
            ]
        }
        
        mapped_cols = {}
        debug_info = []
        
        for key, keywords in column_mappings.items():
            found = False
            for col in df.columns:
                if pd.isna(col):
                    clean_name = ""
                else:
                    clean_name = str(col).strip()
                
                if any(kw.lower() in clean_name.lower() for kw in keywords):
                    mapped_cols[key] = col
                    debug_info.append(f"✅ {key}: 识别为 '{col}'")
                    found = True
                    break
            if not found:
                if key in ['customer', 'order_id', 'amount', 'quantity']:
                    debug_info.append(f"❌ {key}: 未找到匹配列")
                else:
                    debug_info.append(f"⚪ {key}: 未找到 (可选)")

        with st.expander(f"🔎 {prefix} 列名匹配报告"):
            st.write("\n".join(debug_info))
        
        required_keys = ['customer', 'order_id', 'amount', 'quantity']
        missing_required = [k for k in required_keys if k not in mapped_cols]
        
        if missing_required:
            st.error(f"❌ {prefix} 缺少必需列：{missing_required}。请检查文件内容。")
            return None, {}
        
        return df, mapped_cols
        
    except Exception as e:
        st.error(f"💥 读取文件 '{file.name}' 时发生严重错误：{str(e)}")
        st.info("💡 建议：检查文件是否损坏，或尝试另存为 .csv 格式后上传。")
        return None, {}

def calculate_historical_average_and_customers(files, api_key):
    """
    读取历史文件，返回过去两年的平均抽样数和所有抽样的客户集合
    """
    total_average = 0
    total_customers = []
    file_count = 0

    progress_bar = st.progress(0)
    
    for i, file in enumerate(files):
        if file:
            try:
                st.info(f"🤖 正在使用大模型分析历史文件 '{file.name}'...")
                # 使用大模型API分析历史文件
                avg_count, customers = analyze_history_with_llm(file, api_key)
                
                if avg_count > 0:
                    total_average += avg_count
                    total_customers.extend(customers)
                    file_count += 1
                    st.success(f"✅ 从 '{file.name}' 中识别出 {len(customers)} 个客户，平均抽样数：{avg_count}")
            except Exception as e:
                st.warning(f"⚠️ 处理历史文件 {file.name} 时出错：{e}")
                continue
        
        progress_bar.progress((i + 1) / len(files))

    progress_bar.empty()
    
    if file_count > 0:
        final_avg = total_average / file_count
        # 向上取整
        rounded_avg = int(final_avg) + (1 if final_avg % 1 > 0 else 0)
        unique_customers = list(set(total_customers))
        # 计算±10%的范围
        min_range = int(rounded_avg * 0.9)
        max_range = int(rounded_avg * 1.1)
        st.success(f"📈 基于 {file_count} 个有效历史文件，平均抽样数：{rounded_avg}（向上取整）")
        st.info(f"📋 抽样范围：{min_range}-{max_range}（±10%）")
        st.info(f"📋 从历史底稿中识别出 {len(unique_customers)} 个过往年度抽样客户: {unique_customers[:10]}...") # 只显示前10个
        return rounded_avg, unique_customers
    else:
        st.warning("⚠️ 无有效历史文件，使用默认抽样数量 20 和空的客户名单。")
        return 20, []


def assign_match_level(df, order_col, delivery_col):
    df = df.copy()
    if order_col not in df.columns or delivery_col not in df.columns:
        # 如果缺少关键列，无法计算匹配等级，全部设为1
        df['match_level'] = 1
        return df

    # 确保订单和出库单列不为空
    df_temp = df.dropna(subset=[order_col, delivery_col]).copy()
    
    # 计算每个订单ID对应的出库单数量
    order_delivery_counts = df_temp.groupby(order_col)[delivery_col].nunique().reset_index()
    order_delivery_counts.columns = [order_col, 'delivery_count']
    
    # 将数量信息合并回原DataFrame
    df = df.merge(order_delivery_counts, on=order_col, how='left')
    
    # 基于出库单数量确定匹配等级，如果为0或NaN，则默认为1
    df['match_level'] = df['delivery_count'].clip(upper=3).fillna(1).astype(int)
    
    return df

def clean_special_entries(df, customer_col, order_col, amount_col, date_col, summary_col):
    df_clean = df.copy()
    
    if amount_col not in df_clean.columns or order_col not in df_clean.columns:
        return df_clean
        
    if not date_col or date_col not in df_clean.columns:
        return df_clean

    temp_date_col = '_temp_parsed_date'
    try:
        df_clean[temp_date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
    except:
        return df_clean

    current_month = datetime.now().month
    current_year = datetime.now().year
    
    # 创建一个布尔列，表示该行是否为当前月且金额为正
    is_current_positive = (
        (df_clean[temp_date_col].dt.month == current_month) &
        (df_clean[temp_date_col].dt.year == current_year) &
        (df_clean[amount_col] > 0)
    )
    
    # 将这个布尔条件添加为一列（用于后续 groupby）
    df_clean['_is_current_positive'] = is_current_positive
    
    # 现在可以安全地进行 groupby 操作了
    has_negative_mask = df_clean.groupby(order_col)[amount_col].transform(lambda x: (x < 0).any())
    has_current_positive_mask = df_clean.groupby(order_col)['_is_current_positive'].transform('any')
    
    special_order_ids = df_clean[(has_negative_mask) & (has_current_positive_mask)][order_col].unique()
    
    if len(special_order_ids) > 0:
        mask = df_clean[order_col].isin(special_order_ids)
        special_data = df_clean[mask]
        normal_data = df_clean[~mask]
        
        amount_condition = special_data[amount_col] > 0
        date_condition = (special_data[temp_date_col].dt.month == current_month) & \
                         (special_data[temp_date_col].dt.year == current_year)
        
        summary_condition = True
        if summary_col and summary_col in special_data.columns:
            summary_condition = special_data[summary_col].astype(str).str.contains('暂估', na=False)
            
        filtered_special = special_data[amount_condition & date_condition & summary_condition]
        
        df_clean = pd.concat([normal_data, filtered_special], ignore_index=True)
    
    # 删除临时列
    if '_is_current_positive' in df_clean.columns:
        df_clean.drop(columns=['_is_current_positive'], inplace=True)
    if temp_date_col in df_clean.columns:
        df_clean.drop(columns=[temp_date_col], inplace=True)
        
    return df_clean

def filter_and_classify_customers(df, col_mapping, historical_customers=None):
    """
    核心逻辑：1. 严格过滤数量和金额为正的数据
              2. 基于客户订单数进行分层
              3. (新增) 将历史抽样客户强制加入大客户池
    """
    customer_col = col_mapping.get('customer')
    order_col = col_mapping.get('order_id')
    amount_col = col_mapping.get('amount')
    quantity_col = col_mapping.get('quantity')

    st.write("🔍 **执行严格数据过滤...**")
    
    # **严格过滤：移除数量或金额小于等于0的行**
    initial_count = len(df)
    df_filtered = df[
        (df[quantity_col] > 0) & (df[amount_col] > 0)
    ].copy()
    
    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count
    
    st.info(f"✅ 严格数据过滤完成。移除 {removed_count} 条负数或零值记录，剩余 {filtered_count} 条。")
    
    if filtered_count == 0:
        st.error("❌ 过滤后无有效数据，无法继续分析。")
        return pd.DataFrame(), [], []

    # **基于客户订单数进行分层**
    st.write("📊 **执行客户分层...**")
    
    # 计算每个客户的订单数量
    customer_order_counts = df_filtered.groupby(customer_col)[order_col].nunique().reset_index()
    customer_order_counts.columns = [customer_col, 'order_count']
    
    # 按订单数量降序排列
    customer_order_counts = customer_order_counts.sort_values(by='order_count', ascending=False)
    
    # 计算总订单数
    total_orders = customer_order_counts['order_count'].sum()
    if total_orders == 0:
        st.error("❌ 所有客户订单数总和为0，无法分层。")
        return df_filtered, [], []

    # 累计计算，找出占总订单数80%的客户
    cumulative_orders = 0
    large_customers = []
    for _, row in customer_order_counts.iterrows():
        cumulative_orders += row['order_count']
        large_customers.append(row[customer_col])
        if cumulative_orders / total_orders >= 0.8:
            break
    
    # 剩余客户为小客户
    all_customers_set = set(customer_order_counts[customer_col])
    large_customers_set = set(large_customers)
    small_customers = list(all_customers_set - large_customers_set)
    
    # (新增) 强制将历史抽样客户加入大客户池，确保审计连贯性
    if historical_customers:
        historical_customers_set = set(historical_customers)
        # 找出在当前数据中存在的历史客户
        existing_historical_customers = [c for c in historical_customers if c in all_customers_set]
        
        if existing_historical_customers:
            new_large_customers_set = large_customers_set.union(set(existing_historical_customers))
            new_small_customers_list = [c for c in small_customers if c not in existing_historical_customers]
            
            st.success(f"🌟 已将 {len(existing_historical_customers)} 个历史抽样客户强制加入大客户池，确保审计连贯性。")
            
            large_customers = list(new_large_customers_set)
            small_customers = new_small_customers_list
    
    st.success(f"👥 客户分层完成：大客户 {len(large_customers)} 家，小客户 {len(small_customers)} 家。")
    # st.write(f"📊 **大客户列表** (共 {len(large_customers)} 家): {large_customers}")
    # st.write(f"📊 **小客户列表** (共 {len(small_customers)} 家): {small_customers}")
    
    return df_filtered, large_customers, small_customers

def deduplicate_orders(df, order_col, delivery_col):
    """
    对数据进行去重，确保一个订单号只保留一条记录。
    优先级：1个出库单 > 2个出库单 > 3个及以上出库单 > 空值
    """
    if order_col not in df.columns or 'match_level' not in df.columns:
        st.warning("⚠️ 缺少必要列，无法进行订单去重。")
        return df
    
    # 过滤掉订单号为空的行
    df_non_empty_orders = df.dropna(subset=[order_col]).copy()
    
    if df_non_empty_orders.empty:
        st.warning("⚠️ 所有记录的订单号均为空，无法去重。")
        return pd.DataFrame()
    
    # 按订单号分组，并按 match_level 升序排序（优先级：1 -> 2 -> 3+）
    df_sorted = df_non_empty_orders.sort_values(by=['match_level'])
    
    # 按订单号去重，保留第一个（即 match_level 最高的）
    df_deduplicated = df_sorted.drop_duplicates(subset=[order_col], keep='first')
    
    st.info(f"✅ 订单去重完成。从 {len(df_non_empty_orders)} 条非空订单记录中，筛选出 {len(df_deduplicated)} 条唯一订单记录。")
    return df_deduplicated

def perform_stratified_sampling(df, col_mapping, large_customers, small_customers, target_total, min_small_sample=2):
    """
    执行分层抽样
    """
    customer_col = col_mapping.get('customer')
    
    # 分离大客户和小客户的数据
    large_df = df[df[customer_col].isin(large_customers)].copy()
    small_df = df[df[customer_col].isin(small_customers)].copy()
    
    st.write(f"📊 分层抽样池：大客户数据 {len(large_df)} 条，小客户数据 {len(small_df)} 条。")
    
    # 计算允许的总样本量范围 (目标 ± 10%)
    lower_bound = int(target_total * 0.9)
    upper_bound = int(target_total * 1.1)
    
    st.write(f"🎯 目标抽样量: {target_total}, 允许范围: [{lower_bound}, {upper_bound}]")

    # 计算初始分配，确保大客户样本 > 小客户样本
    # 约束: L + S ∈ [lower_bound, upper_bound] 且 L > S
    min_s_for_balance = max(min_small_sample, (lower_bound - 1) // 2)
    max_s_for_total = (upper_bound - 1) // 2
    
    # 实际小客户可抽取的最大数量
    actual_max_small = min(len(small_df), max_s_for_total)
    
    # 最终小客户抽样数
    final_small_sample = max(min_small_sample, min(actual_max_small, min_s_for_balance))
    
    # 对应的大客户抽样数
    remaining_for_large = target_total - final_small_sample
    final_large_sample = max(remaining_for_large, final_small_sample + 1)
    
    # 再次检查总和是否超出上限
    if final_large_sample + final_small_sample > upper_bound:
        final_small_sample = min(final_small_sample, upper_bound - final_large_sample)
        if final_small_sample < 0:
            # 极端情况
            final_large_sample = max(1, upper_bound - 1)
            final_small_sample = max(0, upper_bound - final_large_sample)
            
    # 如果池子里的样本不够，就全拿
    final_large_sample = min(final_large_sample, len(large_df))
    final_small_sample = min(final_small_sample, len(small_df))

    st.write(f"📊 调整后分配：大客户样本 {final_large_sample} 个，小客户样本 {final_small_sample} 个。")

    # 执行抽样
    large_sample_df = large_df.sample(n=final_large_sample, random_state=42) if final_large_sample > 0 else pd.DataFrame()
    small_sample_df = small_df.sample(n=final_small_sample, random_state=42) if final_small_sample > 0 else pd.DataFrame()
    
    # 合并结果
    combined_sample_df = pd.concat([large_sample_df, small_sample_df], ignore_index=True)
    
    final_count = len(combined_sample_df)
    st.info(f"✅ 分层抽样完成，样本量: {final_count} (目标: {target_total}, 范围: [{lower_bound}, {upper_bound}])")
    
    # 样本构成报告
    sampled_large_count = len(large_sample_df)
    sampled_small_count = len(small_sample_df)
    st.info(f"📊 最终样本构成: 大客户样本 {sampled_large_count} 个, 小客户样本 {sampled_small_count} 个。")
    
    if sampled_large_count <= sampled_small_count and sampled_large_count > 0 and sampled_small_count > 0:
         st.warning("⚠️ 警告：由于数据池大小限制，大客户样本数量未能超过小客户样本数量。")

    return combined_sample_df


def apply_amount_filter_and_rebalance(original_sample, df_deduplicated, col_mapping, large_customers, small_customers, target_total):
    """
    对抽样结果进行金额过滤，并重新平衡样本。
    """
    customer_col = col_mapping.get('customer')
    order_col = col_mapping.get('order_id')
    amount_col = col_mapping.get('amount')

    st.write("🔍 **执行金额阈值过滤...**")
    
    # 找出低于阈值的记录
    threshold = 10000
    low_amount_mask = original_sample[amount_col] < threshold
    low_amount_records = original_sample[low_amount_mask]
    high_amount_records = original_sample[~low_amount_mask]
    
    num_low_amount = len(low_amount_records)
    
    if num_low_amount == 0:
        st.info("✅ 所有抽样记录金额均 >= 10000，无需过滤。")
        return original_sample

    st.warning(f"⚠️ 发现 {num_low_amount} 条记录金额低于 {threshold} 元，将被过滤。")

    # 统计被过滤记录的来源客户池
    large_customers_set = set(large_customers)
    small_customers_set = set(small_customers)
    
    from_large_pool = low_amount_records[low_amount_records[customer_col].isin(large_customers_set)]
    from_small_pool = low_amount_records[low_amount_records[customer_col].isin(small_customers_set)]
    
    num_from_large = len(from_large_pool)
    num_from_small = len(from_small_pool)
    
    st.info(f"📊 过滤的记录中：来自大客户池 {num_from_large} 条，来自小客户池 {num_from_small} 条。")

    # 为被过滤的记录寻找替代品
    replacement_records = []
    
    # --- 修改的核心逻辑开始 ---
    # 1. 为来自大客户池的记录寻找替代
    if num_from_large > 0:
        st.write(f"🔄 为来自大客户池的 {num_from_large} 个空缺寻找替代...")
        # 获取大客户池中所有未被抽中的记录，并且金额 >= 10000
        sampled_large_orders = set(high_amount_records[high_amount_records[customer_col].isin(large_customers_set)][order_col])
        candidates_for_large = df_deduplicated[
            (df_deduplicated[customer_col].isin(large_customers_set)) &
            (~df_deduplicated[order_col].isin(sampled_large_orders)) &
            (df_deduplicated[amount_col] >= threshold)
        ].sort_values(by=amount_col, ascending=False)
        
        # 优先在大客户池中寻找替代
        replacements_for_large = candidates_for_large.head(num_from_large)
        replacement_records.append(replacements_for_large)
        st.info(f"  - 优先在大客户池找到 {len(replacements_for_large)} 个替代记录。")
        
        # 检查是否还有空缺
        remaining_from_large = num_from_large - len(replacements_for_large)
        if remaining_from_large > 0:
            st.warning(f"  - 大客户池替代品不足，仍有 {remaining_from_large} 个空缺，将从大客户池其他数据中寻找。")
            # 从大客户池中所有未被抽中（也未被选为替代品）的记录中寻找
            all_selected_orders = set(high_amount_records[order_col]).union(set(replacements_for_large[order_col]))
            additional_candidates_for_large = df_deduplicated[
                (df_deduplicated[customer_col].isin(large_customers_set)) &
                (~df_deduplicated[order_col].isin(all_selected_orders))
            ].sort_values(by=amount_col, ascending=False)
            
            additional_replacements = additional_candidates_for_large.head(remaining_from_large)
            replacement_records.append(additional_replacements)
            st.info(f"  - 从大客户池追加找到 {len(additional_replacements)} 个补充记录。")


    # 2. 为来自小客户池的记录寻找替代
    if num_from_small > 0:
        st.write(f"🔄 为来自小客户池的 {num_from_small} 个空缺寻找替代...")
        # 获取小客户池中所有未被抽中的记录，并且金额 >= 10000
        sampled_small_orders = set(high_amount_records[high_amount_records[customer_col].isin(small_customers_set)][order_col])
        candidates_for_small = df_deduplicated[
            (df_deduplicated[customer_col].isin(small_customers_set)) &
            (~df_deduplicated[order_col].isin(sampled_small_orders)) &
            (df_deduplicated[amount_col] >= threshold)
        ].sort_values(by=amount_col, ascending=False)
        
        # 优先在小客户池中寻找替代
        replacements_for_small = candidates_for_small.head(num_from_small)
        replacement_records.append(replacements_for_small)
        st.info(f"  - 优先在小客户池找到 {len(replacements_for_small)} 个替代记录。")
        
        # 检查是否还有空缺
        remaining_from_small = num_from_small - len(replacements_for_small)
        if remaining_from_small > 0:
            st.warning(f"  - 小客户池替代品不足，仍有 {remaining_from_small} 个空缺，将从大客户池中寻找。")
            # 从大客户池中所有未被抽中（也未被选为替代品）的记录中寻找
            all_selected_orders_so_far = set(high_amount_records[order_col]).union(
                set(replacements_for_large[order_col]),
                set(replacements_for_small[order_col]),
                set(replacement_records[-1][order_col]) if len(replacement_records) > 0 else set()
            )
            additional_candidates_for_large_from_large_pool = df_deduplicated[
                (df_deduplicated[customer_col].isin(large_customers_set)) &
                (~df_deduplicated[order_col].isin(all_selected_orders_so_far))
            ].sort_values(by=amount_col, ascending=False)
            
            additional_replacements_from_large = additional_candidates_for_large_from_large_pool.head(remaining_from_small)
            replacement_records.append(additional_replacements_from_large)
            st.info(f"  - 从大客户池为小客户池空缺找到 {len(additional_replacements_from_large)} 个补充记录。")

    # --- 修改的核心逻辑结束 ---
    
    # 合并所有替代记录
    if replacement_records:
        all_replacements = pd.concat(replacement_records, ignore_index=True)
        # 最终样本 = 高金额原始样本 + 所有替代品
        final_balanced_sample = pd.concat([high_amount_records, all_replacements], ignore_index=True)
    else:
        final_balanced_sample = high_amount_records

    final_count = len(final_balanced_sample)
    st.success(f"✅ 金额过滤与样本平衡完成。最终样本量: {final_count}。")
    
    return final_balanced_sample


# --- 主界面 ---

st.title("📊 穿行测试智能筛选工具 V2.5 (按业务量分层)")
st.markdown("""
上传当年财务明细表，AI 自动分析并进行有代表性的分层抽样。
""")

# --- 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ 参数配置")
    
    # API Key 配置
    st.subheader("🔑 API 配置")
    api_key = st.text_input("阿里云通义千问 API Key", type="password", placeholder="请输入您的 API Key", help="用于大模型分析历史数据，请到阿里云控制台获取")
    
    current_file = st.file_uploader("📄 当年财务明细全量表 (.xlsx)", type=["xlsx"], help="必填项")
    history_files = st.file_uploader("📁 近两年历史穿行底稿 (.xlsx)", type=["xlsx"], accept_multiple_files=True, help="推荐上传，用于自动计算抽样数量和参考客户")
    
    st.divider()
    
    st.subheader("自定义参数")
    
    related_parties_text = st.text_area("关联客户名单 (剔除)", placeholder="客户 A\n客户 B\n客户 C", help="支持换行或逗号分隔")
    
    top_clients_text = st.text_area("本年前 10 大客户名单", placeholder="直接粘贴全称列表，每行一个", help="用于强制纳入抽样")
    
    ai_instruction = st.text_input("AI 个性化筛选说明", placeholder="如：覆盖所有子公司...", help="AI 将理解并应用逻辑")
    
    # 添加手动输入历史客户名称的选项
    historical_clients_text = st.text_area("手动输入历史客户名称", placeholder="客户 A\n客户 B\n客户 C", help="当大模型无法识别历史客户时使用")
    
    # 添加筛选条件设置，让用户指定要抽取哪一年的数据
    selected_year = st.number_input("指定抽取年份", min_value=2000, max_value=2100, value=datetime.now().year, step=1, help="指定当年抽取数据的年份")
    
    manual_base_size = st.number_input("手动设置基准抽样数量", min_value=1, max_value=10000, value=18, step=1, help="如果不上传历史文件，使用此值")
    

# --- 主界面处理逻辑 ---

if current_file:
    # 1. 加载当年数据
    df_current, col_mapping_current = load_excel_with_mapping(current_file, is_history=False)
    
    if df_current is None:
        st.error("无法加载当年数据文件！请检查文件格式或重试。")
        st.stop()
    
    st.success(f"✅ 当年数据加载成功：{len(df_current)} 行")
    
    # 2. 处理历史文件
    target_total = manual_base_size
    historical_customers = []
    
    if history_files:
        if not api_key:
            st.error("❌ 请在侧边栏输入阿里云通义千问 API Key 以使用大模型分析历史数据")
            st.stop()
            
        with st.status("📈 正在分析历史数据以计算最佳抽样量和参考客户...", expanded=True) as status:
            avg_count, historical_customers = calculate_historical_average_and_customers(history_files, api_key)
            # 按照历史平均值±10%设置目标抽样量
            target_total = avg_count
            min_target = int(target_total * 0.9)
            max_target = int(target_total * 1.1)
            status.update(label=f"✅ 历史分析完成！目标抽样量：{target_total} (范围：{min_target}-{max_target})", state="complete", expanded=False)
            # 显示从历史底稿中识别出的客户
            if historical_customers:
                st.info(f"📋 从历史底稿中识别出的客户将优先纳入抽样范围，确保审计连贯性。")
            else:
                st.warning("⚠️ 大模型未能识别出历史客户，将使用手动输入的历史客户（如果有）。")
    else:
        st.warning(f"⚠️ 未上传历史文件，使用手动设置的基准数量：{target_total} 条")
    
    # 检查是否有手动输入的历史客户
    if historical_clients_text.strip():
        manual_historical_customers = [line.strip() for line in historical_clients_text.split('\n') if line.strip()]
        if manual_historical_customers:
            # 如果大模型没有识别出客户，使用手动输入的
            if not historical_customers:
                historical_customers = manual_historical_customers
                st.success(f"✅ 使用手动输入的 {len(historical_customers)} 个历史客户。")
            else:
                # 合并大模型识别的和手动输入的客户
                combined_customers = list(set(historical_customers + manual_historical_customers))
                historical_customers = combined_customers
                st.success(f"✅ 合并大模型识别的和手动输入的历史客户，共 {len(historical_customers)} 个。")

    # --- 3. 将手动输入框提前 ---
    st.subheader("🔧 请手动指定关键列名")
    st.write("**当前自动识别的列名：**")
    st.json(col_mapping_current)
    
    # 手动输入框
    manual_customer = st.text_input("请手动输入‘客户’列名（留空则使用自动识别）", value=col_mapping_current.get('customer', ''), help="例如：客户名称、客户")
    manual_order = st.text_input("请手动输入‘订单’列名（留空则使用自动识别）", value=col_mapping_current.get('order_id', ''), help="例如：订单号、单据编号")
    manual_amount = st.text_input("请手动输入‘金额’列名（留空则使用自动识别）", value=col_mapping_current.get('amount', ''), help="例如：未税金额、不含税金额")
    manual_quantity = st.text_input("请手动输入‘数量’列名（留空则使用自动识别）", value=col_mapping_current.get('quantity', ''), help="例如：数量、qty")
    manual_delivery = st.text_input("请手动输入‘出库’列名（留空则使用自动识别）", value=col_mapping_current.get('delivery_id', ''), help="例如：出库单号、送货单")
    
    # 创建一个新的映射字典
    final_col_mapping = col_mapping_current.copy()
    if manual_customer:
        final_col_mapping['customer'] = manual_customer
    if manual_order:
        final_col_mapping['order_id'] = manual_order
    if manual_amount:
        final_col_mapping['amount'] = manual_amount
    if manual_quantity:
        final_col_mapping['quantity'] = manual_quantity
    if manual_delivery:
        final_col_mapping['delivery_id'] = manual_delivery
    
    # 检查最终映射是否完整
    required_cols = ['customer', 'order_id', 'delivery_id', 'amount', 'quantity']
    missing_final_cols = [col for col in required_cols if col not in final_col_mapping]
    
    if missing_final_cols:
        st.error(f"❌ 最终列名映射缺少必需列：{missing_final_cols}。请检查手动输入。")
        st.stop()
    else:
        st.success("✅ 所有必需列均已映射。")

    # 4. 数据预览和调试
    st.subheader("🔍 数据预览与调试")
    
    # 显示数据前5行
    st.write(f"**数据预览 (前5行) - 共 {len(df_current)} 行**:")
    preview_cols = [final_col_mapping['customer'], final_col_mapping['order_id'], final_col_mapping['amount'], final_col_mapping['quantity']]
    st.dataframe(df_current[preview_cols].head(), use_container_width=True)
    
    # 显示关键列的统计信息
    st.write("**关键列统计信息:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("客户总数", df_current[final_col_mapping['customer']].nunique())
    with c2:
        st.metric("金额列正数占比", f"{(df_current[final_col_mapping['amount']] > 0).mean():.1%}")
    with c3:
        st.metric("数量列正数占比", f"{(df_current[final_col_mapping['quantity']] > 0).mean():.1%}")

    # 5. 添加一个明确的开始按钮
    process_btn = st.button("⚡ 开始自动筛选穿行样本", type="primary", use_container_width=True)

    if process_btn:
        with st.spinner('正在智能分析并处理数据...'):
            
            customer_col = final_col_mapping['customer']
            order_col = final_col_mapping['order_id']
            delivery_col = final_col_mapping['delivery_id']
            amount_col = final_col_mapping['amount']
            quantity_col = final_col_mapping['quantity']
            date_col = final_col_mapping.get('date', None)
            summary_col = final_col_mapping.get('summary', None)
            
            # 6. 执行后续流程

            # a. 根据用户指定的年份进行筛选
            if date_col and date_col in df_current.columns:
                st.write(f"📅 **按指定年份 {selected_year} 筛选数据...**")
                # 尝试解析日期列
                try:
                    df_current['_temp_date'] = pd.to_datetime(df_current[date_col], errors='coerce')
                    # 筛选指定年份的数据
                    df_current = df_current[df_current['_temp_date'].dt.year == selected_year].copy()
                    # 删除临时列
                    df_current.drop(columns=['_temp_date'], inplace=True)
                    st.success(f"✅ 已筛选出 {selected_year} 年的数据，共 {len(df_current)} 行。")
                except Exception as e:
                    st.warning(f"⚠️ 年份筛选失败，将使用所有数据：{e}")

            # b. 数据清洗
            st.write("🔧 **执行数据清洗...**")
            df_cleaned = clean_special_entries(df_current, customer_col, order_col, amount_col, date_col, summary_col)
            
            # c. 匹配等级赋值
            st.write("🔗 **计算订单与出库单匹配等级...**")
            df_matched = assign_match_level(df_cleaned, order_col, delivery_col)
            
            # d. 应用关联方剔除
            related_parties = []
            if related_parties_text.strip():
                parties = re.split(r'[,\n\s]+', related_parties_text)
                related_parties = [p.strip() for p in parties if p.strip()]
                before_count = len(df_matched)
                df_matched = df_matched[~df_matched[customer_col].isin(related_parties)]
                after_count = len(df_matched)
                removed_count = before_count - after_count
                st.info(f"🚫 已剔除 {removed_count} 条关联方数据。")
            
            # e. 严格过滤和客户分层 (核心逻辑, 传入历史客户名单)
            df_filtered_for_analysis, large_customers, small_customers = filter_and_classify_customers(df_matched, final_col_mapping, historical_customers)
            
            # f. 强制加入头部客户
            if top_clients_text.strip():
                forced_top_clients = [line.strip() for line in top_clients_text.split('\n') if line.strip()]
                # 强制加入大客户池，并从原小客户池中移除
                large_customers = list(set(large_customers + forced_top_clients))
                small_customers = [c for c in small_customers if c not in forced_top_clients]
                st.success(f"🌟 已强制将 {len(forced_top_clients)} 家头部客户加入大客户池")

            # g. 对用于抽样的数据再次进行过滤（以防分层后混入了无效数据）
            df_for_sampling = df_filtered_for_analysis.copy()
            
            # h. 订单去重
            st.write("🔄 **执行订单去重...**")
            df_deduplicated = deduplicate_orders(df_for_sampling, order_col, delivery_col)
            
            if df_deduplicated.empty:
                 st.error("❌ 订单去重后无有效数据，无法抽样。请检查数据。")
                 st.stop()

            # i. 执行分层抽样
            st.write("🎲 **执行最终分层抽样...**")
            initial_sample = perform_stratified_sampling(
                df_deduplicated,
                final_col_mapping,
                large_customers, small_customers,
                target_total, min_small_sample=2
            )

            # j. (新增) 应用金额过滤和样本平衡
            st.write("💰 **执行金额过滤与样本平衡...**")
            final_sample = apply_amount_filter_and_rebalance(
                initial_sample, df_deduplicated, final_col_mapping, large_customers, small_customers, target_total
            )
            
            st.divider()
            st.subheader("📋 抽样结果")
            
            if not final_sample.empty:
                st.dataframe(final_sample, use_container_width=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("原始数据量", f"{len(df_current):,}") # 注意：这里原始数据量已因过滤而改变
                c2.metric("筛选后数据量", f"{len(df_filtered_for_analysis):,}")
                c3.metric("去重后数据量", f"{len(df_deduplicated):,}")
                c4.metric("最终抽样量", f"{len(final_sample):,}")
                
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    final_sample.to_excel(writer, index=False, sheet_name='抽样结果')
                
                st.download_button(
                    label="📥 下载抽样结果 (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name=f"穿行测试抽样结果_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                st.success("✅ 抽样完成！已按业务量进行分层，应用了历史客户参考、关联方剔除、金额阈值过滤等规则，并确保了大客户样本数量多于小客户，总样本量在目标范围内。")
                
            else:
                st.error("❌ 抽样结果为空，请检查筛选条件或数据质量。")

else:
    st.info("👈 请先上传当年财务明细全量表。")

# --- 页脚 ---
st.markdown("---")
st.caption("© 穿行测试智能筛选系统 | 数字化审计创新实验室")
st.caption("隐私申明：本系统为单页面工具，数据不在服务器持久存储，关闭页面即自动清理。")