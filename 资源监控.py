import time
import psutil
import argparse
import subprocess
import os
import sys
import csv
from datetime import datetime
import statistics

# 尝试导入 pynvml 库
try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


class ExpertMonitor:
    def __init__(self, interval=0.5, gpu_id=0, log_file="system_metrics.csv"):
        self.interval = interval
        self.gpu_id = gpu_id
        self.log_file = log_file
        self.use_pynvml = HAS_PYNVML
        self.running = True

        # 数据历史容器
        self.history = {
            'timestamp': [],
            'cpu_max_core': [],  # 最忙的那个核的利用率
            'cpu_avg': [],  # 所有核平均
            'cpu_per_core': [],  # 所有核的详细数据 (List[List])
            'ram_used_gb': [],
            'swap_used_gb': [],
            'gpu_util': [],
            'gpu_mem_used_gb': [],
            'gpu_power_watts': [],  # 显卡功耗
            'gpu_temp_c': []  # 显卡温度
        }

        # 硬件静态信息
        self.cpu_count = psutil.cpu_count(logical=True)
        self.ram_total_gb = psutil.virtual_memory().total / 1024 ** 3
        self.gpu_name = "Unknown GPU"
        self.gpu_mem_total_gb = 0
        self.gpu_power_limit = 0

        # 初始化 GPU
        self._init_gpu()

        # 初始化 CSV
        self._init_csv()

    def _init_gpu(self):
        if self.use_pynvml:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

                # 获取名称
                name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(name, bytes): name = name.decode('utf-8')
                self.gpu_name = name

                # 获取显存总量
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.gpu_mem_total_gb = mem_info.total / 1024 ** 3

                # 获取功耗上限 (毫瓦 -> 瓦)
                try:
                    self.gpu_power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
                except:
                    self.gpu_power_limit = 0

                print(
                    f"[Info] GPU 驱动加载成功: {self.gpu_name} | VRAM: {self.gpu_mem_total_gb:.1f} GB | TDP: {self.gpu_power_limit} W")
            except Exception as e:
                print(f"[Warn] pynvml 初始化出错 ({e})，降级使用 nvidia-smi")
                self.use_pynvml = False

        if not self.use_pynvml:
            # Fallback 简单获取显存总量
            try:
                res = subprocess.check_output(
                    ['nvidia-smi', f'--id={self.gpu_id}', '--query-gpu=memory.total', '--format=csv,nounits,noheader'])
                self.gpu_mem_total_gb = float(res.decode('utf-8').strip()) / 1024
                print(f"[Info] 使用 nvidia-smi 监控 | VRAM: {self.gpu_mem_total_gb:.1f} GB")
            except:
                print("[Error] 无法检测到 NVIDIA GPU，请检查驱动！")

    def _init_csv(self):
        # 创建 CSV 头部
        header = ['Time', 'CPU_Avg(%)', 'CPU_Max_Core(%)', 'RAM_Used(GB)', 'Swap_Used(GB)',
                  'GPU_Util(%)', 'GPU_Mem(GB)', 'GPU_Power(W)', 'GPU_Temp(C)']
        # 添加每个核心的列
        for i in range(self.cpu_count):
            header.append(f'Core_{i}(%)')

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[Info] 详细性能日志将写入: {self.log_file}")

    def get_gpu_dynamic_stats(self):
        """获取 GPU 实时动态数据"""
        util, mem_gb, power, temp = 0, 0, 0, 0
        if self.use_pynvml:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024 ** 3
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                return util, mem, power, temp
            except:
                return 0, 0, 0, 0
        else:
            # nvidia-smi fallback (仅获取利用率和显存，功耗温度解析较慢暂略)
            try:
                res = subprocess.check_output(
                    ['nvidia-smi', f'--id={self.gpu_id}', '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,nounits,noheader'],
                    encoding='utf-8'
                )
                u_str, m_str = res.strip().split(',')
                return float(u_str), float(m_str) / 1024, 0, 0
            except:
                return 0, 0, 0, 0

    def step(self):
        # 1. CPU (获取每个核心)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        cpu_avg = sum(cpu_per_core) / len(cpu_per_core)
        cpu_max = max(cpu_per_core)

        # 2. Memory
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ram_gb = mem.used / 1024 ** 3
        swap_gb = swap.used / 1024 ** 3

        # 3. GPU
        gpu_u, gpu_m, gpu_p, gpu_t = self.get_gpu_dynamic_stats()

        # 4. 记录历史
        now_str = datetime.now().strftime("%H:%M:%S")
        self.history['timestamp'].append(now_str)
        self.history['cpu_avg'].append(cpu_avg)
        self.history['cpu_max_core'].append(cpu_max)
        self.history['cpu_per_core'].append(cpu_per_core)
        self.history['ram_used_gb'].append(ram_gb)
        self.history['swap_used_gb'].append(swap_gb)
        self.history['gpu_util'].append(gpu_u)
        self.history['gpu_mem_used_gb'].append(gpu_m)
        self.history['gpu_power_watts'].append(gpu_p)
        self.history['gpu_temp_c'].append(gpu_t)

        # 5. 写入 CSV
        row = [now_str, f"{cpu_avg:.1f}", f"{cpu_max:.1f}", f"{ram_gb:.2f}", f"{swap_gb:.2f}",
               f"{gpu_u:.1f}", f"{gpu_m:.2f}", f"{gpu_p:.1f}", f"{gpu_t:.1f}"]
        row.extend([f"{x:.1f}" for x in cpu_per_core])

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        return cpu_avg, cpu_max, ram_gb, swap_gb, gpu_u, gpu_m, gpu_p, gpu_t

    def print_dashboard(self, c_avg, c_max, ram, swap, g_u, g_m, g_p, g_t):
        # 构造一个高密度的实时显示条
        # 显存条
        vram_pct = (g_m / self.gpu_mem_total_gb) * 100 if self.gpu_mem_total_gb > 0 else 0
        # 红色警报
        cpu_color = "!" if c_max > 95 else " "
        gpu_color = "!" if g_u > 95 else " "
        mem_color = "!" if vram_pct > 90 else " "

        msg = (
            f"\rCPU[{cpu_color}]: Avg {c_avg:4.1f}% | MaxCore {c_max:4.1f}% || "
            f"RAM: {ram:4.1f}GB (Swap {swap:3.1f}GB) || "
            f"GPU[{gpu_color}]: {g_u:4.1f}% | VRAM[{mem_color}]: {g_m:4.1f}GB | {g_p:3.0f}W | {g_t:2.0f}C "
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    def run(self, duration=None):
        print("\n" + "=" * 90)
        print(f"🚀 专家模式监控启动 | Log: {self.log_file}")
        print(f"   监测: {self.cpu_count} CPU Cores | {self.gpu_name}")
        print("=" * 90)

        # 首次调用 psutil.cpu_percent 需要间隔
        psutil.cpu_percent(percpu=True)
        time.sleep(0.5)

        start_time = time.time()
        try:
            while True:
                if duration and (time.time() - start_time > duration):
                    break
                stats = self.step()
                self.print_dashboard(*stats)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n\n🛑 监控停止，正在分析数据...")

        self.generate_deep_analysis(time.time() - start_time)

    def generate_deep_analysis(self, duration):
        if not self.history['cpu_avg']: return

        def safe_mean(l):
            return statistics.mean(l) if l else 0

        def safe_max(l):
            return max(l) if l else 0

        def safe_percentile(l, p):
            l_sorted = sorted(l)
            return l_sorted[int(len(l) * p)] if l else 0

        print("\n" + "=" * 90)
        print(f"🔬 深度性能瓶颈诊断报告 (时长: {duration / 60:.1f} min)")
        print("=" * 90)

        # --- 1. CPU 微观分析 ---
        print("【CPU 核心分析】")
        cpu_max_list = self.history['cpu_max_core']
        avg_max_core = safe_mean(cpu_max_list)
        peak_max_core = safe_max(cpu_max_list)

        print(f"  • 最忙核心平均负载: {avg_max_core:.1f}%")
        print(f"  • 最忙核心峰值负载: {peak_max_core:.1f}%")

        # 分析大小核/负载均衡
        # 计算每个核心在整个过程中的平均负载
        core_avgs = []
        num_samples = len(self.history['cpu_per_core'])
        for i in range(self.cpu_count):
            core_load_sum = sum(sample[i] for sample in self.history['cpu_per_core'])
            core_avgs.append(core_load_sum / num_samples)

        # 简单的 ASCII 柱状图展示核心负载
        print("  • 核心负载分布图 (Core 0 -> Core N):")
        print("    ", end="")
        for load in core_avgs:
            if load > 80:
                char = "█"  # 极高
            elif load > 60:
                char = "▓"
            elif load > 40:
                char = "▒"
            elif load > 20:
                char = "░"
            else:
                char = "_"
            print(char, end="")
        print(f" (Max: {max(core_avgs):.1f}%, Min: {min(core_avgs):.1f}%)")

        # 诊断
        if avg_max_core > 90 and safe_mean(self.history['cpu_avg']) < 60:
            print("  ⚠️  [严重瓶颈] 单核性能受限！")
            print("     现象：总 CPU 利用率不高，但某个核心长期跑满。")
            print("     原因：Python GIL锁、物理引擎单线程计算、或任务被错误调度到了单个核心。")
            print("     建议：检查代码中的环境并行方式；如果用了大量 num_envs，确认是否都在同一个主进程等待。")
        elif avg_max_core > 90:
            print("  ⚠️  [瓶颈] CPU 计算力不足 (全核满载)。建议减少 num_envs 或升级 CPU。")
        else:
            print("  ✅ CPU 性能充裕，未发现明显瓶颈。")

        print("-" * 50)

        # --- 2. GPU 深度分析 ---
        print("【GPU 性能分析】")
        g_util = self.history['gpu_util']
        g_mem = self.history['gpu_mem_used_gb']
        g_pwr = self.history['gpu_power_watts']

        avg_util = safe_mean(g_util)
        p95_mem = safe_percentile(g_mem, 0.95)

        print(f"  • 平均计算负载: {avg_util:.1f}% (P95: {safe_percentile(g_util, 0.95):.1f}%)")
        print(f"  • 显存峰值使用: {safe_max(g_mem):.2f} GB / {self.gpu_mem_total_gb:.1f} GB")
        if self.gpu_power_limit > 0:
            avg_pwr = safe_mean(g_pwr)
            print(
                f"  • 平均功耗: {avg_pwr:.1f} W (TDP {self.gpu_power_limit:.0f} W, 占比 {avg_pwr / self.gpu_power_limit * 100:.1f}%)")

        # 诊断
        if avg_util < 15:
            print("  📉 [闲置] GPU 严重饥饿。")
            print("     原因：CPU 收集数据太慢，或者数据传输(PCIe)太慢。")
            print("     解决：大幅增加 num_envs，或检查 CPU 单核瓶颈。")
        elif avg_util > 95:
            print("  🔥 [瓶颈] GPU 算力瓶颈。模型训练/渲染已经满负荷。")
        elif 85 < p95_mem / self.gpu_mem_total_gb * 100:
            print("  ⚠️  [危险] 显存接近极限，可能随时 OOM。请勿增加 Batch Size。")
        else:
            print("  ✅ GPU 负载健康。")

        print("-" * 50)

        # --- 3. 内存与 Swap ---
        max_swap = safe_max(self.history['swap_used_gb'])
        if max_swap > 0.1:
            print(f"  ⚠️  [警告] 检测到 Swap 使用 ({max_swap:.1f} GB)！")
            print("     这会极大地拖慢训练速度。系统物理内存不足。")
        else:
            print(f"  ✅ 内存充足，无 Swap 交换。")

        print("=" * 90)
        print(f"详细原始数据已保存至: {self.log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔(秒)")
    args = parser.parse_args()

    monitor = ExpertMonitor(interval=args.interval)
    monitor.run()  # 按 Ctrl+C 结束并生成报告