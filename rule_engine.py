import json
import pandas as pd
import numpy as np


def get_theoretical_threshold_by_radius(approx_value_raws):
    """根据半径获取理论阈值"""
    if approx_value_raws <= 10:
        return 83
    elif approx_value_raws <= 20:
        return 41
    elif approx_value_raws <= 30:
        return 27
    elif approx_value_raws <= 40:
        return 20
    elif approx_value_raws <= 50:
        return 16
    elif approx_value_raws <= 60:
        return 13
    elif approx_value_raws <= 70:
        return 11
    elif approx_value_raws <= 80:
        return 10
    elif approx_value_raws <= 90:
        return 9
    elif approx_value_raws <= 100:
        return 8
    elif approx_value_raws <= 110:
        return 7
    elif approx_value_raws <= 120:
        return 6
    elif approx_value_raws <= 130:
        return 6
    elif approx_value_raws <= 140:
        return 5
    else:
        return 5


def load_rules_from_json(json_file_path):
    """从JSON文件加载分类规则"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            rules_config = json.load(f)
        return rules_config
    except Exception as e:
        print(f"加载规则文件失败: {str(e)}")
        return None


def save_rules_to_json(rules_config, json_file_path):
    """保存分类规则到JSON文件"""
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(rules_config, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存规则文件失败: {str(e)}")
        return False


def check_channel_combination(row, combination):
    """检查通道组合条件"""
    d_has_value = row.get('DW1O_TotalSNR', 0) > 0
    j_has_value = row.get('DW2O_TotalSNR', 0) > 0
    p_has_value = row.get('DN1O_TotalSNR', 0) > 0
    
    if combination == "D_only":
        return d_has_value and not j_has_value and not p_has_value
    elif combination == "J_only":
        return not d_has_value and j_has_value and not p_has_value
    elif combination == "P_only":
        return not d_has_value and not j_has_value and p_has_value
    elif combination == "D_and_J":
        return d_has_value and j_has_value and not p_has_value
    elif combination == "D_and_P":
        return d_has_value and not j_has_value and p_has_value
    elif combination == "J_and_P":
        return not d_has_value and j_has_value and p_has_value
    elif combination == "D_and_J_and_P":
        return d_has_value and j_has_value and p_has_value
    else:
        return True  # 无通道限制


def evaluate_condition(row, condition, thresholds):
    """评估单个条件"""
    feature = condition['feature']
    operator = condition['operator']
    value = condition['value']
    use_threshold = condition.get('use_threshold', False)
    
    # 获取特征值
    feature_value = row.get(feature, 0)
    
    # 如果使用阈值调整
    if use_threshold:
        threshold_key = None
        if 'DW1O_TotalSNR' in feature or 'DW1O_MainRowSNR' in feature:
            threshold_key = 'snr_adjustment'
        elif 'DW2O_TotalSNR' in feature or 'DW2O_MainRowSNR' in feature:
            threshold_key = 'snr_adjustment'
        elif 'DN1O_TotalSNR' in feature:
            threshold_key = 'snr_adjustment'
        elif 'DW1O_MaxOrg' in feature or 'DW1O_Peak' in feature:
            threshold_key = 'dw1o_peak_adjustment'
        elif 'DW2O_MaxOrg' in feature or 'DW2O_Peak' in feature:
            threshold_key = 'dw2o_peak_adjustment'
        
        if threshold_key and threshold_key in thresholds:
            value = value + thresholds[threshold_key]
    
    # 根据操作符评估
    if operator == '>':
        return feature_value > value
    elif operator == '>=':
        return feature_value >= value
    elif operator == '<':
        return feature_value < value
    elif operator == '<=':
        return feature_value <= value
    elif operator == '==':
        return feature_value == value
    elif operator == '!=':
        return feature_value != value
    else:
        return False


def evaluate_logic_expression(row, conditions, logic_expression, thresholds):
    """评估复杂的逻辑表达式
    
    Args:
        row: 数据行
        conditions: 条件列表
        logic_expression: 逻辑表达式字符串，例如 "1 && (2 || 3 || 4) && (!5)"
        thresholds: 阈值配置
    
    Returns:
        bool: 表达式评估结果
    """
    if not logic_expression or not conditions:
        return False
    
    # 首先评估所有条件，建立条件索引到布尔值的映射
    condition_results = {}
    for idx, condition in enumerate(conditions):
        condition_id = condition.get('condition_id', idx + 1)
        result = evaluate_condition(row, condition, thresholds)
        condition_results[str(condition_id)] = result
    
    # 替换表达式中的条件ID为实际的布尔值
    expression = logic_expression
    
    # 将条件ID替换为True/False
    for cond_id in sorted(condition_results.keys(), key=lambda x: -len(x)):  # 从长到短排序，避免替换错误
        expression = expression.replace(str(cond_id), str(condition_results[cond_id]))
    
    # 替换逻辑运算符
    expression = expression.replace('&&', ' and ')
    expression = expression.replace('||', ' or ')
    expression = expression.replace('!', ' not ')
    
    try:
        # 安全评估表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return bool(result)
    except Exception as e:
        print(f"逻辑表达式评估错误: {expression}, 错误: {str(e)}")
        return False


def apply_rules_from_json(row, rules_config):
    """应用JSON配置的规则到单行数据"""
    thresholds = rules_config.get('thresholds', {})
    rules = rules_config.get('rules', [])
    default_return = rules_config.get('default_return', 10002)
    
    # 按rule_id排序规则
    rules = sorted(rules, key=lambda x: x.get('rule_id', 999))
    
    # 遍历所有规则
    for rule in rules:
        # 跳过未启用的规则
        if not rule.get('enabled', True):
            continue
        
        # 检查通道组合
        channel_combination = rule.get('channel_combination', None)
        if channel_combination:
            if not check_channel_combination(row, channel_combination):
                continue
        
        # 评估所有条件
        conditions = rule.get('conditions', [])
        
        if not conditions:
            continue
        
        # 检查是否使用复杂逻辑表达式
        logic_expression = rule.get('logic_expression', None)
        
        if logic_expression:
            # 使用复杂逻辑表达式
            if evaluate_logic_expression(row, conditions, logic_expression, thresholds):
                return rule.get('return_value', default_return)
        else:
            # 使用简单逻辑（AND/OR）
            logic = rule.get('logic', 'AND')
            
            # 评估每个条件
            condition_results = [evaluate_condition(row, cond, thresholds) for cond in conditions]
            
            # 根据逻辑组合条件结果
            if logic == 'AND':
                if all(condition_results):
                    return rule.get('return_value', default_return)
            elif logic == 'OR':
                if any(condition_results):
                    return rule.get('return_value', default_return)
    
    # 如果没有规则匹配，返回默认值
    return default_return


def process_dataframe_with_rules(df, rules_config):
    """使用规则配置处理整个数据框"""
    df['nDefectType'] = df.apply(lambda row: apply_rules_from_json(row, rules_config), axis=1)
    return df
