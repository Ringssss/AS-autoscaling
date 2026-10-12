"""Read-only progress renderer; does not submit or modify serving requests."""
import datetime
import json
from pathlib import Path
import time

root=Path(__file__).parent/'results'/'campaign-20260928-r3'
while True:
    try:
        state=json.loads((root/'status.json').read_text())
        spec=state.get('current',{})
        trial=Path(state.get('current_directory',root))
        heartbeat=json.loads((trial/'heartbeat.json').read_text()) if (trial/'heartbeat.json').exists() else {}
        lines=['# AgentX 三节点 baseline 实时进度','',
            '更新时间：'+datetime.datetime.now().astimezone().isoformat(),'',
            f"- 调度器状态：{state['state']}；PID：{state['pid']}",
            f"- 已完成正式轨迹：{state['completed']} / {state['total']}",
            f"- 当前：{spec.get('scenario','—')}，全局 BS={spec.get('bs','—')}，{spec.get('policy','—')}",
            f"- 当前请求完成：{heartbeat.get('completed',0)} / {heartbeat.get('expected','—')}",
            f"- 当前轨迹运行：{heartbeat.get('elapsed_seconds',0):.1f} 秒",
            f"- 三个引擎客户端在途数：{heartbeat.get('live','—')}",
            f"- 待 admission/依赖完成：{heartbeat.get('pending','—')}",
            f"- 状态准备计数：{heartbeat.get('preparation_counts',{})}", '',
            '正式矩阵保留选定 AgentX 轨迹原始输入/输出长度；跨轮和扩容不清缓存。',
            '三轮不是独立重复；BS 是客户端在途上限，不是保证达到的 GPU batch。',
            '启动状态文件中的 healthy 数组仅是启动期旧探测值；本视图用调度状态和运行心跳展示进度。', '',
            '[已完成结果](PROGRESS.md) · [协议](../../PROTOCOL.zh.md)']
        if 'error' in state:lines+=['','失败原因：'+state['error']]
        if state['state']=='running' and not Path('/proc/'+str(state['pid'])).exists():
            lines+=['','进程已消失；状态可能陈旧，不能视为仍在运行。']
            (root/'LIVE.md').write_text('\n'.join(lines)+'\n');break
        (root/'LIVE.md').write_text('\n'.join(lines)+'\n')
        if state['state'] in ['failed','complete']:break
    except (FileNotFoundError,json.JSONDecodeError):pass
    time.sleep(10)
