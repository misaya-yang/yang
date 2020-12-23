import numpy as np
import pandas as pd
from ortools.graph import pywrapgraph
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings("ignore")

""" read data"""
buyer_df = pd.read_csv(r'D:\aDatas\buyer.csv', encoding='gb2312', dtype=str, na_values='nan')
seller_df = pd.read_csv(r'D:\aDatas\seller.csv', encoding='gb2312', dtype=str, na_values='nan')
print('read data done!')
""" preprocessing data"""
# convert dtype
buyer_df[['平均持仓时间', '购买货物数量']] = buyer_df[['平均持仓时间', '购买货物数量']].astype(int)
seller_df[['货物数量（张）']] = seller_df[['货物数量（张）']].astype(int)
# combine the hope key and value
buyer_df['hope1'] = [
    f'{key}_{val}' for key,
                       val in zip(
        buyer_df['第一意向'],
        buyer_df['值'])]
buyer_df['hope2'] = [
    f'{key}_{val}' for key,
                       val in zip(
        buyer_df['第二意向'],
        buyer_df['值.1'])]
buyer_df['hope3'] = [
    f'{key}_{val}' for key,
                       val in zip(
        buyer_df['第三意向'],
        buyer_df['值.2'])]
buyer_df['hope4'] = [
    f'{key}_{val}' for key,
                       val in zip(
        buyer_df['第四意向'],
        buyer_df['值.3'])]
buyer_df['hope5'] = [
    f'{key}_{val}' for key,
                       val in zip(
        buyer_df['第五意向'],
        buyer_df['值.4'])]
buyer_df.drop(['第一意向', '第二意向', '第三意向', '第四意向', '第五意向', '值',
               '值.1', '值.2', '值.3', '值.4'], axis=1, inplace=True)

goods_group_cfg = {'货物数量（张）': 'sum',
                   '仓库': 'last',
                   '品牌': 'last',
                   '产地': 'last',
                   '年度': 'last',
                   '等级': 'last',
                   '类别': 'last'
                   }

GOODS_ATTR = seller_df.groupby('货物编号').agg(goods_group_cfg).drop(
    '货物数量（张）', axis=1).to_dict(
    orient='index')

attr_cols = ['仓库', '品牌', '产地', '年度', '等级', '类别']
for c in attr_cols:
    seller_df[c] = f'{c}_' + seller_df[c].str[:]

cf_seller = seller_df[seller_df['品种'] == 'CF']
sr_seller = seller_df[seller_df['品种'] == 'SR']
cf_seller_goods = cf_seller.groupby('货物编号').agg(goods_group_cfg)
cf_seller_goods.fillna('nan', inplace=True)
sr_seller_goods = sr_seller.groupby('货物编号').agg(goods_group_cfg)
sr_seller_goods.fillna('nan', inplace=True)
cf_buyer = buyer_df[buyer_df['品种'] == 'CF']
sr_buyer = buyer_df[buyer_df['品种'] == 'SR']

print('preprocess data done!')


def matching_source_demand(
        start_nodes,
        end_nodes,
        capacities,
        costs,
        supplies,
        source_list,
        demand_list,
        source_name='goods',
        demand_name='buyer'):
    # create max-flow-min-cost network
    costs = [int(i) for i in costs]
    capacities = [int(i) for i in capacities]

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(
            start_nodes[i], end_nodes[i], capacities[i], costs[i])
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])
    st = min_cost_flow.SolveMaxFlowWithMinCost()
    all_hope_matching_df = pd.DataFrame(
        [],
        columns=[
            source_name,
            demand_name,
            'qty',
            'cap',
            'cost'])
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        all_hope_matching_df[source_name] = [
            min_cost_flow.Tail(i) for i in range(
                min_cost_flow.NumArcs())]
        all_hope_matching_df[demand_name] = [
            min_cost_flow.Head(i) for i in range(
                min_cost_flow.NumArcs())]
        all_hope_matching_df['qty'] = [
            min_cost_flow.Flow(i) for i in range(
                min_cost_flow.NumArcs())]
        all_hope_matching_df['cap'] = [
            min_cost_flow.Capacity(i) for i in range(
                min_cost_flow.NumArcs())]
        all_hope_matching_df['cost'] = all_hope_matching_df['qty'] * \
                                       [min_cost_flow.UnitCost(i) for i in range(min_cost_flow.NumArcs())]
        hope_matching_df = all_hope_matching_df[all_hope_matching_df['qty'] > 0]
    else:
        print('There was an issue with the min cost flow input.')
    hope_matching_df[source_name] = hope_matching_df[source_name].apply(
        lambda x: source_list[x])
    hope_matching_df[demand_name] = hope_matching_df[demand_name].apply(
        lambda x: demand_list[x - len(source_list)])
    hope_matching_df.sort_values(by=[demand_name, 'cost'], inplace=True)
    return hope_matching_df


"""
match the first hope：
source: goods_id
demand: hope_id
"""


def first_hope_cost(goods_attr, hope):
    # first hope cost
    hope_split = hope.split('_')
    hope_k = hope_split[0]
    hope_v = hope_split[1]
    if hope_v == 'nan':
        return 1
    elif goods_attr.get(hope_k) != hope_v:
        return 33
    else:
        return 0


def get_goods_id_list(df):
    # calculate goods_id_list
    return sorted(df.index.tolist())


def get_1st_hope_list(df):
    # calculate 1st hope list
    first_hope_list = df['hope1'].unique().tolist()
    return sorted(first_hope_list)


def get_flow_cost_1st_hope(goods_df, buyer_df, goods_list, first_hope_list):
    # calculate cost and flow
    start_nodes = []
    end_nodes = []
    cost_all = []
    flow_all = []
    global GOODS_ATTR
    for g in goods_list:
        for h in first_hope_list:
            cost = first_hope_cost(GOODS_ATTR[g], h)
            cost_all.append(cost)
    flow_single = goods_df['货物数量（张）'].tolist()
    for f in flow_single:
        flow_all += [f] * len(first_hope_list)
    for i, g in enumerate(goods_list):
        start_nodes += [i] * len(first_hope_list)
    for _ in range(len(goods_list)):
        end_nodes += [i + len(goods_list) for i in range(len(first_hope_list))]
    return cost_all, flow_all, start_nodes, end_nodes


def get_supply_1st_hope(goods_df, buyer_df):
    # calculate supply
    source = goods_df['货物数量（张）'].sort_index().tolist()
    demand = buyer_df[['购买货物数量', 'hope1']].groupby(
        'hope1').sum().sort_values(by='hope1')['购买货物数量'].tolist()
    out = []
    out += source
    out += [-x for x in demand]
    return out


def prepare_network_arcs_1st_hope(goods_df, buyer_df):
    # prepare nodes for 1st hope network
    # print('starting to calculate the arcs...')
    goods_df.sort_index(inplace=True)
    buyer_df.sort_values(by='hope1', inplace=True)
    goods_list = get_goods_id_list(goods_df)
    hope_list = get_1st_hope_list(buyer_df)
    cost_all, flow_all, start, end = get_flow_cost_1st_hope(
        goods_df, buyer_df, goods_list, hope_list)
    supply = get_supply_1st_hope(goods_df, buyer_df)
    return start, end, flow_all, cost_all, supply, goods_list, hope_list


def proc_matching_1st_hope(
        goods_df,
        buyer_df,
        source_name='goods',
        demand_name='hope1'):
    start_nodes, end_nodes, capacities, costs, supplies, goods_list, hope_list = prepare_network_arcs_1st_hope(
        goods_df, buyer_df)
    hope_matching_df = matching_source_demand(
        start_nodes,
        end_nodes,
        capacities,
        costs,
        supplies,
        goods_list,
        hope_list,
        source_name=source_name,
        demand_name=demand_name)
    return hope_matching_df


"""
handle hope1_met groups
"""


def get_buyer_list(buyer_df):
    buyer_list = buyer_df['买方客户'].unique().tolist()
    return sorted(buyer_list)


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def get_flow_cost_rest_hopes(
        goods_df,
        buyer_df,
        goods_list,
        buyer_list,
        seller_attr_df,
        cf=True):
    start_nodes = []
    end_nodes = []
    cost_all = []
    flow_all = []
    if cf:
        cost_list = [27, 20, 13, 7]
        hope_num_list = ['hope2', 'hope3', 'hope4', 'hope5']
    else:
        cost_list = [30, 20, 10]
        hope_num_list = ['hope2', 'hope3', 'hope4']

    for goods_chunk in tqdm(list(chunks(goods_list, 100))):
        good_attr_df = seller_attr_df.loc[goods_chunk][attr_cols].sort_index(
        ).reset_index()
        good_attr_df = pd.concat(
            [good_attr_df] * len(buyer_list),
            ignore_index=True).sort_values('货物编号')
        rep_buyer_df = pd.concat(
            [buyer_df] * len(goods_chunk),
            ignore_index=True)
        mix_hope_df = pd.concat(
            [rep_buyer_df[hope_num_list], good_attr_df[attr_cols]], axis=1)
        del good_attr_df, rep_buyer_df
        gc.collect()
        for col, cost in zip(hope_num_list, cost_list):
            mix_hope_df[f'{col}_mask'] = ~(mix_hope_df.drop(
                col, 1).isin(mix_hope_df[col]).any(1)) * cost
        cost_chunk = mix_hope_df[[f'{col}_mask' for col in hope_num_list]].sum(
            axis=1).values.tolist()
        del mix_hope_df
        gc.collect()
        cost_all += cost_chunk
    flow_single = goods_df['qty'].tolist()
    for f in flow_single:
        flow_all += [f] * len(buyer_list)
    del flow_single
    gc.collect()
    for i, g in enumerate(goods_list):
        start_nodes += [i] * len(buyer_list)
    for _ in range(len(goods_list)):
        end_nodes += [i + len(goods_list) for i in range(len(buyer_list))]
    return cost_all, flow_all, start_nodes, end_nodes


def get_supply_rest_hopes(goods_df, buyer_df):
    source = goods_df['qty'].sort_index().tolist()
    demand = buyer_df[['购买货物数量', '买方客户']].sort_values(by='买方客户')[
        '购买货物数量'].tolist()
    out = []
    out += source
    out += [-x for x in demand]
    return out


def prepare_network_arcs_rest_hope(
        goods_df_h,
        buyer_df_h,
        seller_attr_df,
        cf=True):
    goods_df_h.sort_values(by='goods', inplace=True)
    buyer_df_h.sort_values(by='买方客户', inplace=True)

    goods_list_h = sorted(goods_df_h['goods'].unique().tolist())
    buyer_list_h = get_buyer_list(buyer_df_h)
    cost_h, flow_h, start_nodes_h, end_nodes_h = get_flow_cost_rest_hopes(
        goods_df_h, buyer_df_h, goods_list_h, buyer_list_h, seller_attr_df, cf=cf)
    supply_h = get_supply_rest_hopes(goods_df_h, buyer_df_h)
    return start_nodes_h, end_nodes_h, flow_h, cost_h, supply_h, goods_list_h, buyer_list_h


def proc_matching_rest_hopes(
        h,
        fisrt_hope_matching_df,
        buyer_df,
        seller_attr_df,
        source_name='goods',
        demand_name='buyer',
        cf=True):
    goods_df_h = fisrt_hope_matching_df[fisrt_hope_matching_df['hope1'] == h][[
        'goods', 'qty']]
    buyer_df_h = buyer_df[buyer_df['hope1'] == h][[
        '买方客户', '购买货物数量', 'hope2', 'hope3', 'hope4', 'hope5']]
    start_nodes, end_nodes, capacities, costs, supplies, goods_list, hope_list = prepare_network_arcs_rest_hope(
        goods_df_h, buyer_df_h, seller_attr_df, cf=cf)
    del goods_df_h, buyer_df_h
    gc.collect()
    hope_matching_df = matching_source_demand(
        start_nodes,
        end_nodes,
        capacities,
        costs,
        supplies,
        goods_list,
        hope_list,
        source_name=source_name,
        demand_name=demand_name)
    return hope_matching_df


"""
handle hope1_met_partial groups
"""


def proc_matching_rest_hopes_with_partial(
        h,
        fisrt_hope_matching_df,
        buyer_df,
        seller_attr_df,
        source_name='goods',
        demand_name='buyer',
        cf=True):
    goods_df_h = fisrt_hope_matching_df[fisrt_hope_matching_df['hope1'] == h][[
        'goods', 'qty', 'cost']]
    goods_df_no_cost_h = goods_df_h[goods_df_h['cost'] == 0]  # no_cost need provided to certain customer
    goods_df_high_cost_h = goods_df_h[goods_df_h['cost'] != 0]
    buyer_df_h = buyer_df[buyer_df['hope1'] == h][[
        '买方客户', '购买货物数量', 'hope2', 'hope3', 'hope4', 'hope5', '平均持仓时间']]
    buyer_df_h.sort_values(by=['平均持仓时间', '购买货物数量'], ascending=[
        False, False], inplace=True)
    buyer_df_h['cum_sum'] = goods_df_no_cost_h['qty'].sum() - \
                            buyer_df_h['购买货物数量'].cumsum()  # check how much buyer can be met by no-cost goods
    if 0 in buyer_df_h['cum_sum'].values.tolist():
        buyer_df_non_cost_h = buyer_df_h[buyer_df_h['cum_sum'] >= 0]
        buyer_df_high_cost_h = buyer_df_h[buyer_df_h['cum_sum'] < 0]
    else:  # need to split some buyer
        buyer_df_non_cost_h = buyer_df_h[buyer_df_h['cum_sum'] > 0]
        buyer_df_high_cost_h_before_split = buyer_df_h[buyer_df_h['cum_sum'] < 0]
        # must be [0] to ensure dataframe format
        to_be_split = buyer_df_high_cost_h_before_split.iloc[[0], :]
        to_be_split_non_cost = to_be_split.copy()
        to_be_split_non_cost['购买货物数量'] = to_be_split['购买货物数量'] + \
                                         to_be_split['cum_sum']
        to_be_split_high_cost = to_be_split.copy()
        to_be_split_high_cost['购买货物数量'] = - to_be_split['cum_sum']
        buyer_df_non_cost_h = pd.concat(
            [buyer_df_non_cost_h, to_be_split_non_cost], axis=0)
        buyer_df_high_cost_h = pd.concat(
            [to_be_split_high_cost, buyer_df_high_cost_h_before_split.iloc[1:, :]], axis=0)
    start_nodes, end_nodes, capacities, costs, supplies, goods_list, hope_list = prepare_network_arcs_rest_hope(
        goods_df_no_cost_h, buyer_df_non_cost_h, seller_attr_df, cf=cf)
    hope_matching_df_non_cost = matching_source_demand(
        start_nodes,
        end_nodes,
        capacities,
        costs,
        supplies,
        goods_list,
        hope_list,
        source_name=source_name,
        demand_name=demand_name)
    start_nodes, end_nodes, capacities, costs, supplies, goods_list, hope_list = prepare_network_arcs_rest_hope(
        goods_df_high_cost_h, buyer_df_high_cost_h, seller_attr_df, cf=cf)
    hope_matching_df_high_cost = matching_source_demand(
        start_nodes,
        end_nodes,
        capacities,
        costs,
        supplies,
        goods_list,
        hope_list,
        source_name=source_name,
        demand_name=demand_name)
    return hope_matching_df_non_cost, hope_matching_df_high_cost


"""
matching seller and buyer
"""


# calcualate seller_id  #cf_seller


def get_seller_id_list_per_goods(seller_df_per_goods):
    return sorted(seller_df_per_goods['卖方客户'].values.tolist())


# calcualate buyer_id


def get_buyer_id_list_per_goods(buyer_df_per_goods):
    return sorted(buyer_df_per_goods['buyer'].values.tolist())


# calculate cost and flow


def get_flow_cost_seller_buyer(
        seller_df_per_goods,
        buyer_df_per_goods,
        seller_list,
        buyer_list):
    start_nodes = []
    end_nodes = []
    cost_all = []
    flow_all = []
    flow_single = seller_df_per_goods['货物数量（张）'].tolist()
    demand_single = buyer_df_per_goods['qty'].tolist()
    for f in flow_single:
        flow_all += [f] * len(buyer_list)
    del flow_single
    gc.collect()
    for d in demand_single:
        cost_all += [d] * len(seller_list)
    cost_all = [i == j for i, j in zip(flow_all, cost_all)]
    cost_all = list(map(int, cost_all))
    cost_all = [1 - i for i in cost_all]
    for i, g in enumerate(seller_list):
        start_nodes += [i] * len(buyer_list)
    for _ in range(len(seller_list)):
        end_nodes += [i + len(seller_list) for i in range(len(buyer_list))]
    return cost_all, flow_all, start_nodes, end_nodes


# calculate supply


def get_supply_seller_buyer(seller_df_per_goods, buyer_df_per_goods):
    source = seller_df_per_goods['货物数量（张）'].tolist()
    demand = buyer_df_per_goods['qty'].tolist()
    out = []
    out += source
    out += [-x for x in demand]
    return out


# prepare nodes for 1st hope network


def prepare_network_arcs_seller_buyer(seller_df_per_goods, buyer_df_per_goods):
    # print('starting to calculate the arcs...')
    seller_df_per_goods.sort_values(by='卖方客户', inplace=True)
    buyer_df_per_goods.sort_values(by='buyer', inplace=True)
    seller_list = get_seller_id_list_per_goods(seller_df_per_goods)
    buyer_list = get_buyer_id_list_per_goods(buyer_df_per_goods)
    cost_all, flow_all, start, end = get_flow_cost_seller_buyer(
        seller_df_per_goods, buyer_df_per_goods, seller_list, buyer_list)
    supply = get_supply_seller_buyer(seller_df_per_goods, buyer_df_per_goods)
    # print('arcs calculation done.')
    return start, end, flow_all, cost_all, supply, seller_list, buyer_list


def proc_matching_seller_buyer(
        g,
        seller_df,
        buyer_df,
        source_name='seller',
        demand_name='buyer'):
    seller_df_per_goods = seller_df[seller_df['货物编号'] == g][[
        '卖方客户', '货物数量（张）']]
    buyer_df_per_goods = buyer_df[buyer_df['goods'] == g][['buyer', 'qty']]
    start_nodes, end_nodes, capacities, costs, supplies, goods_list, hope_list = prepare_network_arcs_seller_buyer(
        seller_df_per_goods, buyer_df_per_goods)
    del seller_df_per_goods, buyer_df_per_goods
    gc.collect()
    # list_print([start_nodes, end_nodes, capacities, costs, supplies])
    matching_df = matching_source_demand(
        start_nodes,
        end_nodes,
        capacities,
        costs,
        supplies,
        goods_list,
        hope_list,
        source_name=source_name,
        demand_name=demand_name)
    matching_df['goods'] = g
    return matching_df


print('starting matching first hope for sr...')
sr_1st_hope_matching_df = proc_matching_1st_hope(
    sr_seller_goods, sr_buyer, source_name='goods', demand_name='hope1')
sr_1st_hope_cost = sr_1st_hope_matching_df.groupby('hope1').agg(
    {'cost': 'sum'}).sort_values('cost', ascending=False).reset_index()
sr_all_hope_met_list = sr_1st_hope_cost[sr_1st_hope_cost['cost'] == 0]['hope1'].to_list(
)
sr_all_hope_met_list.append('nan_nan')
sr_partial_hope_met_list = [
    i for i in sr_1st_hope_cost['hope1'].to_list() if i not in sr_all_hope_met_list]
print('done.')

print('starting matching hope_met group for sr...')
sr_all_match_df = pd.DataFrame()
for h in tqdm(sr_all_hope_met_list):
    hope_matching_df = proc_matching_rest_hopes(
        h,
        sr_1st_hope_matching_df,
        sr_buyer,
        sr_seller_goods,
        source_name='goods',
        demand_name='buyer',
        cf=False)
    if sr_all_match_df.shape[0] == 0:
        sr_all_match_df = hope_matching_df
    else:
        sr_all_match_df = pd.concat(
            [sr_all_match_df, hope_matching_df], axis=0)
print('done.')

print('starting matching partial_hope_met group for sr...')
for h in tqdm(sr_partial_hope_met_list):
    hope_matching_non_cost_df, hope_matching_high_cost_df = proc_matching_rest_hopes_with_partial(
        h, sr_1st_hope_matching_df, sr_buyer, sr_seller_goods, source_name='goods', demand_name='buyer', cf=False)
    sr_all_match_df = pd.concat(
        [sr_all_match_df, hope_matching_non_cost_df], axis=0)
    sr_all_match_df = pd.concat(
        [sr_all_match_df, hope_matching_high_cost_df], axis=0)
print('done')

print('starting matching seller_buyer for sr...')
sr_seller_buyer_match = pd.DataFrame()
for g in tqdm(sr_seller_goods.index):
    hope_matching_df = proc_matching_seller_buyer(
        g,
        sr_seller,
        sr_all_match_df,
        source_name='seller',
        demand_name='buyer')
    if sr_seller_buyer_match.shape[0] == 0:
        sr_seller_buyer_match = hope_matching_df
    else:
        sr_seller_buyer_match = pd.concat(
            [sr_seller_buyer_match, hope_matching_df], axis=0)
print('done')

del hope_matching_df, sr_all_match_df, hope_matching_non_cost_df, hope_matching_high_cost_df, sr_1st_hope_matching_df
gc.collect()
print(f'size of match is {sr_seller_buyer_match.shape}')
print('starting merging result for sr...')
attr_cols = ['仓库', '品牌', '产地', '年度', '等级', '类别']
right_df = sr_seller_goods.reset_index()
right_df['品种'] = 'SR'
sr_seller_buyer_match = pd.merge(left=sr_seller_buyer_match,
                                 right=right_df[['货物编号', '品种'] + attr_cols],
                                 left_on='goods',
                                 right_on='货物编号',
                                 how='left')

hope_num_list = ['hope1', 'hope2', 'hope3', 'hope4']
sr_seller_buyer_match = pd.merge(left=sr_seller_buyer_match,
                                 right=sr_buyer[['买方客户'] + hope_num_list],
                                 left_on='buyer',
                                 right_on='买方客户',
                                 how='left')
print('starting calculate hope-list result for sr...')
for i, col in tqdm(enumerate(hope_num_list)):
    sr_seller_buyer_match[f'{col}_mask'] = sr_seller_buyer_match[attr_cols].isin(
        sr_seller_buyer_match[col]).any(1).replace({True: str(i + 1), False: ''})

sr_seller_buyer_match['对应意向顺序'] = sr_seller_buyer_match["hope1_mask"] + sr_seller_buyer_match["hope2_mask"] + \
                                  sr_seller_buyer_match["hope3_mask"] + sr_seller_buyer_match["hope4_mask"]
sr_seller_buyer_match['对应意向顺序'] = sr_seller_buyer_match['对应意向顺序'].replace({"": "0"}).str.join('-')

sr_seller_buyer_match['仓库'] = sr_seller_buyer_match['仓库'].str.lstrip('仓库_')
sr_seller_buyer_match.rename(columns={"seller": "卖方客户", "qty": "分配货物数量"}, inplace=True)

result_cols = ['买方客户', '卖方客户', '品种', '货物编号', '仓库', '分配货物数量', '对应意向顺序']
drop_cols = [i for i in sr_seller_buyer_match.columns if i not in result_cols]
sr_seller_buyer_match.drop(drop_cols, axis=1, inplace=True)
print('saving result for sr...')
sr_seller_buyer_match[result_cols].to_csv('sr_result.csv', index=False, encoding='gb2312')
print('*' * 30)
print('sr done!')
print('*' * 30)

print('starting matching first hope for cf...')
cf_1st_hope_matching_df = proc_matching_1st_hope(
    cf_seller_goods, cf_buyer, source_name='goods', demand_name='hope1')
cf_1st_hope_cost = cf_1st_hope_matching_df.groupby('hope1').agg(
    {'cost': 'sum'}).sort_values('cost', ascending=False).reset_index()
cf_all_hope_met_list = cf_1st_hope_cost[cf_1st_hope_cost['cost'] == 0]['hope1'].to_list(
)
cf_all_hope_met_list.append('nan_nan')
cf_partial_hope_met_list = [
    i for i in cf_1st_hope_cost['hope1'].to_list() if i not in cf_all_hope_met_list]
print('done.')

print('starting matching hope_met group for cf...')
cf_all_match_df = pd.DataFrame()
for h in tqdm(cf_all_hope_met_list):
    hope_matching_df = proc_matching_rest_hopes(
        h,
        cf_1st_hope_matching_df,
        cf_buyer,
        cf_seller_goods,
        source_name='goods',
        demand_name='buyer',
        cf=True)
    if cf_all_match_df.shape[0] == 0:
        cf_all_match_df = hope_matching_df
    else:
        cf_all_match_df = pd.concat(
            [cf_all_match_df, hope_matching_df], axis=0)
print('done.')

print('starting matching partial_hope_met group for cf...')
for h in tqdm(cf_partial_hope_met_list):
    hope_matching_non_cost_df, hope_matching_high_cost_df = proc_matching_rest_hopes_with_partial(
        h, cf_1st_hope_matching_df, cf_buyer, cf_seller_goods, source_name='goods', demand_name='buyer', cf=True)
    cf_all_match_df = pd.concat(
        [cf_all_match_df, hope_matching_non_cost_df], axis=0)
    cf_all_match_df = pd.concat(
        [cf_all_match_df, hope_matching_high_cost_df], axis=0)
print('done')

print('starting matching seller_buyer for cf...')
cf_seller_buyer_match = pd.DataFrame()
for g in tqdm(cf_seller_goods.index):
    matching_df = proc_matching_seller_buyer(
        g,
        cf_seller,
        cf_all_match_df,
        source_name='seller',
        demand_name='buyer',
    )
    if cf_seller_buyer_match.shape[0] == 0:
        cf_seller_buyer_match = matching_df
    else:
        cf_seller_buyer_match = pd.concat(
            [cf_seller_buyer_match, matching_df], axis=0)
print('done')

del matching_df, cf_all_match_df, hope_matching_non_cost_df, hope_matching_high_cost_df, cf_1st_hope_matching_df
gc.collect()

print('starting merging result for cf...')
right_df = cf_seller_goods.reset_index()
right_df['品种'] = 'CF'
cf_seller_buyer_match = pd.merge(left=cf_seller_buyer_match,
                                 right=right_df[['货物编号', '品种'] + attr_cols],
                                 left_on='goods',
                                 right_on='货物编号',
                                 how='left')

hope_num_list = ['hope1', 'hope2', 'hope3', 'hope4', 'hope5']
cf_seller_buyer_match = pd.merge(left=cf_seller_buyer_match,
                                 right=cf_buyer[['买方客户'] + hope_num_list],
                                 left_on='buyer',
                                 right_on='买方客户',
                                 how='left')
print('starting calculate hope-list result for cf...')
for i, col in tqdm(enumerate(hope_num_list)):
    cf_seller_buyer_match[f'{col}_mask'] = cf_seller_buyer_match[attr_cols].isin(
        cf_seller_buyer_match[col]).any(1).replace({True: str(i + 1), False: ''})

cf_seller_buyer_match['对应意向顺序'] = cf_seller_buyer_match["hope1_mask"] + cf_seller_buyer_match["hope2_mask"] + \
                                  cf_seller_buyer_match["hope3_mask"] + cf_seller_buyer_match["hope4_mask"] + \
                                  cf_seller_buyer_match["hope5_mask"]
cf_seller_buyer_match['对应意向顺序'] = cf_seller_buyer_match['对应意向顺序'].replace({"": "0"}).str.join('-')

cf_seller_buyer_match['仓库'] = cf_seller_buyer_match['仓库'].str.lstrip('仓库_')
cf_seller_buyer_match.rename(columns={"seller": "卖方客户", "qty": "分配货物数量"}, inplace=True)

drop_cols = [i for i in cf_seller_buyer_match.columns if i not in result_cols]
cf_seller_buyer_match.drop(drop_cols, axis=1, inplace=True)
print('saving result for cf...')
cf_seller_buyer_match[result_cols].to_csv('cf_result.csv', index=False, encoding='gb2312')

final_match_df = pd.concat(
    [cf_seller_buyer_match[result_cols], sr_seller_buyer_match[result_cols]], axis=0, ignore_index=True)

# del cf_seller_buyer_match, sr_seller_buyer_match
# gc.collect()

final_match_df[result_cols].to_csv('result.csv', index=False, encoding='gb2312')
final_match_df[result_cols].to_csv('result.txt', index=False, encoding='gb2312')
