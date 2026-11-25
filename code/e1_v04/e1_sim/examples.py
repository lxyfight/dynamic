"""Small reproducible I0 integration examples, not a policy experiment grid."""
import json
from random import Random
from .initializer import construct_i0, public_delivery_input, verify_plan


def run_i0_examples(package, output):
    network = package["networks"]["fixed_two_city"]
    single = next(c for c in package["cases"] if c["id"] == "E01")["input"]
    examples = [("empty", [], single["drones"]), ("single_E01", single["orders"], single["drones"])]
    for count, seed in ((6, 17), (12, 29)):
        rng = Random(seed)
        orders = []
        for i in range(count):
            city = i % 2 + 1
            orders.append(dict(id=f"D{i+1:03d}", kind="D", weight=rng.randint(6, 19), release=0, deadline=660,
                               city=f"C{city}", xy=[20*city+rng.randint(-28, 28)/4,
                                                    rng.choice([-1, 1])*rng.randint(3, 32)/4],
                               eligible_stations=[f"L{city}+", f"L{city}-"]))
        drones = [dict(id=f"{station}_U{j+1}", home=station) for station in sorted(network["stations"])
                  for j in range(package["configuration"]["drones_per_station"])]
        examples.append((f"deliveries_{count}_seed_{seed}", orders, drones))
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, orders, drones in examples:
        problem = public_delivery_input(package["configuration"], network, orders, drones)
        result = construct_i0(problem)
        (output / (name+"_input.json")).write_text(json.dumps(problem, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
        path = output / (name+"_result.json")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
        row = dict(example=name, deliveries=len(orders), status=result["status"])
        if result["status"] == "VERIFIED":
            reloaded = json.loads(path.read_text(encoding="utf8"))
            check = verify_plan(problem, reloaded["plan"])
            row.update(serialized_replay_passed=check["terminal"], plan_hash=result["plan_hash"],
                       net=check["ledger"]["net"], truck_trips=len(reloaded["plan"]["replay_input"]["trucks"]),
                       active_drones=len({a["drone"] for a in reloaded["plan"]["assignments"]}),
                       last_event=reloaded["plan"]["replay_input"]["stop_time"],
                       construction_seconds=result["construction_seconds"], validation_seconds=result["validation_seconds"])
            lines = [f"{name} 初始配送方案", "I0-basic 单订单构造，无动态取件、车趟合并或局部改进。", ""]
            for a in reloaded["plan"]["assignments"]:
                lines += [f"{a['order']}：{a['quantity']} kg，站点 {a['station']}，卡车 {a['truck']}，无人机 {a['drone']}",
                          f"  卡车0发车；{float(a['truck_arrival']):.6f}到站；{float(a['replenishment_ready']):.6f}卸完；{float(a['truck_return']):.6f}返仓。",
                          f"  无人机{float(a['drone_load']):.6f}装货；{float(a['takeoff']):.6f}起飞；{float(a['service_complete']):.6f}服务完成；{float(a['drone_return']):.6f}返站。"]
            lines += ["", f"配送基准净收益：{row['net']:.6f}元。", "本表仅作显示；重放使用JSON中保留精度的十进制时刻。"]
            (output/(name+"_方案.txt")).write_text("\n".join(lines)+"\n",encoding="utf8")
        else:
            row["reason"] = result.get("reason")
        summaries.append(row)
    (output/"summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return summaries
