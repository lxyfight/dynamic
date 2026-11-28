"""Write a source-grounded Chinese report after both action comparisons finish."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from fit_paired_chain import cluster_stats


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def average_metric(report, name, metric):
    cases=defaultdict(list); layouts=defaultdict(list)
    for r in report['states']:
        cases[(r['spec']['layout_id'],r['case'])].append(r['models'][name][metric])
    for (layout,case),values in cases.items():
        layouts[layout].append(statistics.mean(values))
    return statistics.mean(statistics.mean(v) for v in layouts.values())


def run(root):
    root=Path(root)
    base=read(root/'base/results.json'); aug=read(root/'augmented/results.json')
    audit=read(root/'audit/paired_campaign_integrity.json')
    if not audit.get('successful'):
        raise ValueError('A complete successful independent bank audit is required before reporting.')
    dev=aug['reports']['dev']
    lines=['配对差分训练与一个下游特征：动作检验报告','',
        '本报告是一次首动作干预、后续统一none的条件评价；不是整日新策略利润。',
        '最终测试布局112891—112894未使用。','',
        '数据划分 / 场景数 / 合格状态数 / 每状态条件未来数 / 状态时刻范围']
    branches=0
    for split in ('train','tune','dev'):
        c=read(root/split/'capture_summary.json'); p=read(root/split/'protocol.json')
        sites=read(root/split/'sites.json'); e=read(root/split/'evaluation_complete.json')
        branches+=e['full_day_branches']
        lines.append('%s / %s / %s / %s / %.6f—%.6f分钟'%(split,c['cases'],len(sites),p['samples'],min(s['at'] for s in sites),max(s['at'] for s in sites)))
        pickups=[len(read(f)['reference']['pickup_records']) for f in (root/split/'capture').glob('*.json')]
        lines.append('  原运营日实际pickup数：%d—%d，均值%.3f；N=80是规模参数，不是固定实现订单总数。'%(min(pickups),max(pickups),statistics.mean(pickups)))
    lines+=['共%d个计划内全天分支结果（含每状态一次原未来核验）。另有28次源轨迹采集；缓存补算重复不作为新增统计样本。'%branches,
        '原未来核验不进入训练、调参及条件收益均值。每样本四动作共享未来；未来不进入在线评分。',
        '开发覆盖：'+json.dumps(dev['case_coverage'],ensure_ascii=False),
        '无合格事件场景不补零；下列均值先按场景内状态等权，再按布局内有合格事件的场景等权，最后按8个布局等权。','',
        '阶段A：原chain → 相同两个特征的配对差分chain；同数据level-chain是归因审计。',
        '阶段B：在相同标签上，仅添加“剩余站点预计有利可图pickup机会数”一维。',
        '新特征允许先揭示再预约等待；净收益资格扣除航次、飞行、额外时间及迟到费用。它是单订单机会代理，不保证多单联合可行。','',
        'dev模型 / 0、2、5、10分钟选择数 / 相对立即走均值[95%探索性区间] / 相对统一首次10分钟均值[区间] / 净增接单 / 额外车辆时间费']
    for name in ('old_chain','level_chain','paired_chain','paired_augmented'):
        m=dev['models'][name]; d=m['layout_inference']['depart']; f=m['layout_inference']['fixed10']
        count=[m['choice_counts'].get(str(h),0) for h in (0,2,5,10)]
        lines.append('%s / %s / %.6f [%.6f, %.6f] / %.6f [%.6f, %.6f] / %.6f / %.6f'%(
            name,','.join(map(str,count)),d['mean'],*d['ci95'],f['mean'],*f['ci95'],
            average_metric(dev,name,'accepted_delta_vs_depart'),average_metric(dev,name,'extra_truck_time_cost_vs_depart')))
        lines.append('  布局级符号翻转单侧p：对立即走 %.6f；对统一10分钟 %.6f；平均省车趟 %.6f。'%(d['signflip_one_sided_p'],f['signflip_one_sided_p'],average_metric(dev,name,'trips_saved_vs_depart')))
    direct=cluster_stats([dev['models']['paired_chain']['layout_means'][l]['depart']-dev['models']['level_chain']['layout_means'][l]['depart'] for l in dev['models']['paired_chain']['layout_means']])
    lines.append('同数据直接对照：paired-chain减level-chain = %.6f，布局级95%%区间[%.6f, %.6f]；尚不能确定差分目标单独优于绝对标签目标。'%(direct['mean'],*direct['ci95']))
    best_rows=[]; best_cases=defaultdict(list); best_layouts=defaultdict(list)
    for r in dev['states']:
        h=min((0,2,5,10),key=lambda h:(-r['actions'][str(h)]['gain_vs_depart']['mean'],h))
        advantage=r['actions'][str(h)]['gain_vs_depart']['mean']-r['actions']['10']['gain_vs_depart']['mean']
        best_rows.append(dict(site=r['site_id'],sample_mean_best_h=h,gain_vs_fixed10=advantage))
        best_cases[(r['spec']['layout_id'],r['case'])].append(advantage)
    for (layout,case),values in best_cases.items(): best_layouts[layout].append(statistics.mean(values))
    optimistic=statistics.mean(statistics.mean(v) for v in best_layouts.values())
    diagnostics=dict(same_data_paired_minus_level=direct,posthoc_sample_best=dict(
        layout_equal_gain_vs_fixed10=optimistic,states=best_rows,
        caveat='Optimistic empirical best on the same futures; not a validated policy or population upper bound; never used for model selection.'))
    (root/'diagnostics.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    counts={h:sum(r['sample_mean_best_h']==h for r in best_rows) for h in (0,2,5,10)}
    lines+=['','事后样本内空间诊断（不用于选模型）：',
        '按32个未来样本均值挑每状态最优动作，0/2/5/10的状态数为%s；比统一10分钟仅多%.6f元/次。'%(counts,optimistic),
        '这是同样本择优的乐观描述，不能当作独立验证收益或真实分布的上界。当前样本对自适应间隔留下的改进空间很小。']
    lines+=['','tune选择、参数及权重（raw coefficient直接乘原始特征；intercept=0）']
    for split in ('tune','dev'):
        states=aug['reports'][split]['states']
        same=sum(r['models']['paired_chain']['chosen_h']==r['models']['paired_augmented']['chosen_h'] for r in states)
        lines.append('%s：配对chain与加一维模型选择相同首动作的状态数=%d/%d；同动作使用同一冻结方案、未来和后续策略，收益自然相同。'%(split,same,len(states)))
    for name in ('level_chain','paired_chain','paired_augmented'):
        model=read(root/'augmented/models'/(name+'.json'))
        lines.append('%s: alpha=%g, tune平均增益=%.6f, raw_coefficients=%s'%(name,model['alpha'],model['selection']['tune_equal_case_gain'],model['raw_coefficients']))
    train_features=read(root/'base/known_features/train.json')['states']
    zeros=sum(all(abs(r['features'][i][j]-r['features'][0][j])<1e-12 for j in (0,1)) for r in train_features for i in (1,2,3))
    lines.append('训练的%d个等待—离站对照中，%d个原chain特征差全零；无截距线性差分模型无法为它们给出非零未来价值差。'%(3*len(train_features),zeros))
    first=aug['reports']['train']['states'][0]
    dj=first['J'][2]-first['J'][0]
    dg=first['actions']['5']['gain_vs_depart']['mean']
    lines+=['','首个训练状态的标签实例（固定取排序第一状态，并非按收益挑选）：',
        '%s，等5分钟：ΔJ=%.6f，16样本平均ΔG=%.6f，学习目标ΔG−ΔJ=%.6f。'%(first['site_id'],dj,dg,dg-dj)]
    lines+=['主候选只按tune冻结：'+aug['primary_family'],
        '动作关结果：'+json.dumps(aug['candidate_gates'][aug['primary_family']],ensure_ascii=False),
        '是否进入整日策略比较：'+str(aug['proceed_to_full_day_model_comparison']),
        '门槛为预先冻结主模型相对立即走与统一首次10分钟的两个布局级95%区间下界均>0。',
        '如未通过，本轮停在动作检验，不将新权重扩展到接拒、分站或航次安排；不声称整日策略优胜。',
        '原chain与重训chain也存在新训练日/数据量变化；同数据level对照帮助检查目标函数作用，仍非随机因果试验。',
        '所有区间/符号翻转均为开发诊断，只有8个布局；状态仍集中在早班，不外推午后/晚间。','',
        '逐布局主候选利润差（元）']
    m=dev['models'][aug['primary_family']]
    for layout,v in m['layout_means'].items(): lines.append(layout+' / '+json.dumps(v,ensure_ascii=False))
    lines+=['','文件入口：PROTOCOL.txt；base/results.json；augmented/results.json；各split/paired和offline_futures；audit/paired_campaign_integrity.json。',
        '每状态已知J、特征、动作评分和逐样本G均保留，可复核差分目标和相同未来配对。',
        '训练目标：(G_h-G_0)-(J_h-J_0)；输入：phi_h-phi_0；train-only RMS缩放、无截距ridge；alpha只由tune选。',
        '执行顺序：base --defer-dev → augmented冻结主family并检验dev → base补齐dev报告。',
        '完整性审计概要：'+json.dumps({k:v for k,v in audit.items() if k in ('status','successful','check_count','failed_checks')},ensure_ascii=False)]
    lines+=['搜索预算为每事件基础候选32、增强8、服务/等待MIP新增组合各32；候选截断统计见完整审计JSON。',
        '没有配置/记录wall-time TimeLimit，不能把该项缺失写成零次搜索截断。所有收益都是这一共同搜索预算和none后续策略下的结果。',
        '差分权重只校准本类首动作间的价值差，不能把其绝对输出解释为已校准的完整未来利润。',
        '本轮的统一none后续策略也限定了结论：首个复核到期后仍由none处理，不能据此宣称已经检验了新模型连续复核时的最优间隔。']
    search={name:dict(epochs=0,truncated_epochs=0,forced_reject_requests=0) for name in ('old_chain','level_chain','paired_chain','paired_augmented')}
    for r in dev['states']:
        for replica in range(32):
            pair=read(root/'dev/paired'/(r['site_id']+'_r%02d.json'%replica))
            for name,totals in search.items():
                metrics=pair['branches'][str(r['models'][name]['chosen_h'])]['continuation_search']
                for key in totals: totals[key]+=metrics[key]
    lines+=['','所选首动作对应的后续搜索：模型 / 事件数 / 触及至少一类候选预算的事件数 / 比例（不含固定首动作）']
    for name,totals in search.items():
        rate=totals['truncated_epochs']/totals['epochs']
        lines.append('%s / %d / %d / %.3f%%'%(name,totals['epochs'],totals['truncated_epochs'],100*rate))
    (root/'selected_search_summary.json').write_text(json.dumps(search,indent=2)+'\n',encoding='utf8')
    lines+=['','下一轮建议：保留配对差分训练与同数据绝对标签对照，暂不把新增特征定为最终方案。',
        '优先扩大开发状态的时段和已知任务余量覆盖，再检查0/2/5/10是否存在可学习的收益差别；明确拟验证的连续复核后续策略。',
        '若研究运营中回收车的等待，还需明确已知pickup装载完成、空车到站等事件是否纳入论文决策时点，不能把本次卸delivery后早班结论直接外推。',
        '保留测试布局继续封存；新权重当前仅适用于本类首个复核动作评分。']
    fit_audit=read(root/'audit/fit_final_integrity.json') if (root/'audit/fit_final_integrity.json').exists() else {}
    recovered=read(root/'audit/supplement_recovery_integrity.json') if (root/'audit/supplement_recovery_integrity.json').exists() else {}
    lines+=['','独立拟合/统计终核：'+json.dumps({k:v for k,v in fit_audit.items() if k in ('successful','check_count','failed_checks')},ensure_ascii=False),
        '计算调度恢复：'+json.dumps({k:v for k,v in recovered.items() if k in ('successful','status','original_scheduler_success','verified_scratch_pairs','failures')},ensure_ascii=False),
        'Windows拒绝硬链接发布的原错误日志保留；已用独立副本原子重命名恢复，补算科学内容与主结果严格一致（仅实际耗时除外）。',
        '源码镜像及依赖：source_snapshot/；运行环境：environment.json；全部文件校验：artifact_hashes.json；复现入口：REPRODUCE.ps1。']
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    return dict(branches=branches,primary=aug['primary_family'],gate=aug['proceed_to_full_day_model_comparison'],report=str(root/'REPORT.txt'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    print(json.dumps(run(p.parse_args().output),ensure_ascii=False))
