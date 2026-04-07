import time
import torch
import os
from collections import defaultdict
import datetime
import pytz


class Monitor:
    """监控器类，用于收集和报告性能和资源使用情况"""

    # 日志级别
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self):
        """初始化监控器"""
        # 统计数据
        self.stats = defaultdict(int)
        # 计时器
        self.timers = {}
        # 内存使用记录
        self.memory_stats = []
        # 启动时间
        self.start_time = time.time()
        # 日志级别
        self.log_level = self.INFO
        # 日志文件句柄
        self.log_file = None
        # 初始化日志文件
        self._init_log_file()

    def _init_log_file(self):
        """初始化日志文件
        
        检测logs文件夹是否存在，若不存在则创建。
        日志文件命名为当前系统时间（格式：yyyy-mm-dd_hh-mm-ss.log）
        """
        # 获取当前工作目录
        cwd = os.getcwd()
        # logs文件夹路径
        logs_dir = os.path.join(cwd, "logs")
        
        # 检测logs文件夹是否存在，若不存在则创建
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print(f"[INFO] 创建日志目录: {logs_dir}")
        
        # 生成日志文件名：yyyy-mm-dd_hh-mm-ss.log（亚洲时区）
        asia_tz = pytz.timezone('Asia/Shanghai')
        timestamp = datetime.datetime.now(asia_tz).strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{timestamp}.log"
        log_filepath = os.path.join(logs_dir, log_filename)
        
        # 打开日志文件
        self.log_file = open(log_filepath, "a", encoding="utf-8")
        print(f"[INFO] 日志文件: {log_filepath}")

    def start_timer(self, name):
        """开始计时
        
        Args:
            name: 计时器名称
        """
        self.timers[name] = time.time()

    def end_timer(self, name):
        """结束计时并记录时间
        
        Args:
            name: 计时器名称
        
        Returns:
            经过的时间（秒）
        """
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            self.stats[f"{name}_time"] += elapsed
            self.stats[f"{name}_count"] += 1
            del self.timers[name]
            return elapsed
        return 0

    def increment(self, name, value=1):
        """增加计数器
        
        Args:
            name: 计数器名称
            value: 增加的值
        """
        self.stats[name] += value

    def record_memory(self, name):
        """记录内存使用情况
        
        Args:
            name: 内存记录点名称
        """
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.memory_stats.append({
                "name": name,
                "time": time.time() - self.start_time,
                "memory": memory,
                "max_memory": max_memory
            })
            self.stats[f"{name}_memory"] = memory
            self.stats[f"{name}_max_memory"] = max_memory

    def get_stats(self):
        """获取统计数据
        
        Returns:
            统计数据字典
        """
        return dict(self.stats)

    def get_memory_stats(self):
        """获取内存使用记录
        
        Returns:
            内存使用记录列表
        """
        return self.memory_stats

    def set_log_level(self, level):
        """设置日志级别
        
        Args:
            level: 日志级别，可选值: DEBUG, INFO, WARNING, ERROR
        """
        self.log_level = level

    def _log(self, level, message):
        """内部日志方法
        
        Args:
            level: 日志级别
            message: 日志消息
        """
        if level >= self.log_level:
            # 使用亚洲时区
            asia_tz = pytz.timezone('Asia/Shanghai')
            timestamp = datetime.datetime.now(asia_tz).strftime("%Y-%m-%d %H:%M:%S")
            level_name = {self.DEBUG: "DEBUG", self.INFO: "INFO", self.WARNING: "WARNING", self.ERROR: "ERROR"}[level]
            log_line = f"[{timestamp}] [{level_name}] {message}"
            # 输出到控制台
            print(log_line)
            # 写入到日志文件
            if self.log_file:
                self.log_file.write(log_line + "\n")
                self.log_file.flush()

    def debug(self, message):
        """打印调试日志
        
        Args:
            message: 日志消息
        """
        self._log(self.DEBUG, message)

    def info(self, message):
        """打印信息日志
        
        Args:
            message: 日志消息
        """
        self._log(self.INFO, message)

    def warning(self, message):
        """打印警告日志
        
        Args:
            message: 日志消息
        """
        self._log(self.WARNING, message)

    def error(self, message):
        """打印错误日志
        
        Args:
            message: 日志消息
        """
        self._log(self.ERROR, message)

    def report(self):
        """生成报告
        
        Returns:
            报告字符串
        """
        report = []
        report.append(f"=== Nano-vLLM 监控报告 ===")
        report.append(f"运行时间: {time.time() - self.start_time:.2f} 秒")
        
        # 性能统计
        report.append("\n性能统计:")
        for key, value in self.stats.items():
            if "_time" in key:
                count = self.stats.get(f"{key.replace('_time', '_count')}", 1)
                report.append(f"{key}: {value:.4f} 秒 (平均: {value/count:.4f} 秒/次)")
            elif "_count" not in key and "_memory" not in key:
                report.append(f"{key}: {value}")
        
        # 内存统计
        report.append("\n内存统计:")
        if self.memory_stats:
            max_memory = max([s["max_memory"] for s in self.memory_stats])
            report.append(f"最大内存使用: {max_memory:.2f} MB")
            
            # 最近的内存使用
            recent = self.memory_stats[-5:]
            report.append("最近内存使用:")
            for s in recent:
                report.append(f"  {s['name']}: {s['memory']:.2f} MB (时间: {s['time']:.2f} 秒)")
        
        report_str = "\n".join(report)
        
        # 将报告也写入日志文件
        if self.log_file:
            self.log_file.write("\n" + report_str + "\n")
            self.log_file.flush()
        
        return report_str
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None


# 全局监控器实例
monitor = Monitor()
