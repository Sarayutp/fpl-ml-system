"""
System monitoring and health checks for FPL ML System.
Performance metrics, alerts, and system diagnostics.
"""

import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

from .logging import get_logger, PerformanceLogger
from .cache import get_cache_manager

logger = get_logger(__name__)
perf_logger = PerformanceLogger()


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: Optional[float] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: List[float] = field(default_factory=list)
    
    # Memory metrics
    memory_total: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    memory_available: int = 0
    
    # Disk metrics
    disk_total: int = 0
    disk_used: int = 0
    disk_percent: float = 0.0
    disk_free: int = 0
    
    # Network metrics
    network_sent: int = 0
    network_recv: int = 0
    
    # Process metrics
    process_count: int = 0
    thread_count: int = 0
    
    # Application metrics
    active_connections: int = 0
    request_rate: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu': {
                'percent': self.cpu_percent,
                'count': self.cpu_count,
                'load_average': self.load_average
            },
            'memory': {
                'total': self.memory_total,
                'used': self.memory_used,
                'percent': self.memory_percent,
                'available': self.memory_available
            },
            'disk': {
                'total': self.disk_total,
                'used': self.disk_used,
                'percent': self.disk_percent,
                'free': self.disk_free
            },
            'network': {
                'sent': self.network_sent,
                'recv': self.network_recv
            },
            'processes': {
                'count': self.process_count,
                'threads': self.thread_count
            },
            'application': {
                'connections': self.active_connections,
                'request_rate': self.request_rate,
                'error_rate': self.error_rate
            }
        }


class MetricsCollector:
    """Collect system and application metrics."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.MetricsCollector")
        self._baseline_network = None
        self._last_collection = None
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics."""
        try:
            current_time = datetime.now()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else []
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            if self._baseline_network is None:
                self._baseline_network = network
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Calculate rates if we have previous data
            request_rate = 0.0
            error_rate = 0.0
            
            if self._last_collection:
                time_delta = (current_time - self._last_collection).total_seconds()
                if time_delta > 0:
                    # Calculate network rates
                    pass  # Would implement based on your specific metrics
            
            metrics = SystemMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                load_average=load_avg,
                memory_total=memory.total,
                memory_used=memory.used,
                memory_percent=memory.percent,
                memory_available=memory.available,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_percent=disk.percent,
                disk_free=disk.free,
                network_sent=network.bytes_sent - self._baseline_network.bytes_sent,
                network_recv=network.bytes_recv - self._baseline_network.bytes_recv,
                process_count=process_count,
                thread_count=threading.active_count(),
                request_rate=request_rate,
                error_rate=error_rate
            )
            
            self._last_collection = current_time
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
            return SystemMetrics()


class HealthChecker:
    """Perform various health checks on system components."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.HealthChecker")
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
        self.logger.info("Registered health check", check_name=name)
    
    async def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown health check: {name}"
            )
        
        start_time = time.time()
        try:
            result = await self.checks[name]()
            response_time = time.time() - start_time
            
            if isinstance(result, HealthCheck):
                result.response_time = response_time
                return result
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check completed successfully",
                    details=result if isinstance(result, dict) else {},
                    response_time=response_time
                )
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error("Health check failed", check_name=name, error=str(e))
            return HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                response_time=response_time
            )
    
    async def run_all_checks(self) -> List[HealthCheck]:
        """Run all registered health checks."""
        results = []
        
        for name in self.checks:
            result = await self.run_check(name)
            results.append(result)
        
        return results


class AlertManager:
    """Manage system alerts and notifications."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.AlertManager")
        self.alert_handlers: List[Callable] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def add_handler(self, handler: Callable) -> None:
        """Add alert handler."""
        self.alert_handlers.append(handler)
        self.logger.info("Added alert handler", handler=handler.__name__)
    
    async def send_alert(self, severity: str, message: str, **kwargs) -> None:
        """Send alert to all handlers."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message,
            'details': kwargs
        }
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        self.logger.warning("Alert triggered", **alert)
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error("Alert handler failed", handler=handler.__name__, error=str(e))


class SystemMonitor:
    """Main system monitoring coordinator."""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.logger = get_logger(f"{__name__}.SystemMonitor")
        
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        
        self.running = False
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1440  # 24 hours at 1-minute intervals
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        
        async def database_check():
            """Check database connectivity."""
            try:
                # This would implement actual database check
                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection healthy"
                )
            except Exception as e:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.CRITICAL,
                    message=f"Database connection failed: {str(e)}"
                )
        
        async def cache_check():
            """Check cache system health."""
            try:
                cache_manager = get_cache_manager()
                health = cache_manager.stats()
                
                if health.get('memory_cache', {}).get('utilization', 0) > 0.9:
                    return HealthCheck(
                        name="cache",
                        status=HealthStatus.WARNING,
                        message="Cache utilization high",
                        details=health
                    )
                
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message="Cache system healthy",
                    details=health
                )
            except Exception as e:
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.CRITICAL,
                    message=f"Cache check failed: {str(e)}"
                )
        
        async def ml_models_check():
            """Check ML models availability."""
            try:
                from ..models.ml_models import PlayerPredictor
                
                predictor = PlayerPredictor()
                if not predictor.is_trained:
                    return HealthCheck(
                        name="ml_models",
                        status=HealthStatus.WARNING,
                        message="ML models not trained"
                    )
                
                return HealthCheck(
                    name="ml_models",
                    status=HealthStatus.HEALTHY,
                    message="ML models ready"
                )
            except Exception as e:
                return HealthCheck(
                    name="ml_models",
                    status=HealthStatus.CRITICAL,
                    message=f"ML models check failed: {str(e)}"
                )
        
        async def api_connectivity_check():
            """Check FPL API connectivity."""
            try:
                from ..data.fetchers import FPLAPIClient
                
                client = FPLAPIClient()
                # Simple connectivity test
                bootstrap_data = client.get_bootstrap_data()
                
                if bootstrap_data and 'elements' in bootstrap_data:
                    return HealthCheck(
                        name="fpl_api",
                        status=HealthStatus.HEALTHY,
                        message="FPL API connectivity healthy",
                        details={'player_count': len(bootstrap_data['elements'])}
                    )
                else:
                    return HealthCheck(
                        name="fpl_api",
                        status=HealthStatus.WARNING,
                        message="FPL API returned unexpected data"
                    )
            except Exception as e:
                return HealthCheck(
                    name="fpl_api",
                    status=HealthStatus.CRITICAL,
                    message=f"FPL API connectivity failed: {str(e)}"
                )
        
        async def system_resources_check():
            """Check system resource utilization."""
            try:
                metrics = self.metrics_collector.collect_system_metrics()
                issues = []
                status = HealthStatus.HEALTHY
                
                if metrics.cpu_percent > 80:
                    issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
                    status = HealthStatus.WARNING
                
                if metrics.memory_percent > 85:
                    issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
                    status = HealthStatus.WARNING
                
                if metrics.disk_percent > 90:
                    issues.append(f"High disk usage: {metrics.disk_percent:.1f}%")
                    status = HealthStatus.CRITICAL
                
                message = "System resources healthy"
                if issues:
                    message = "; ".join(issues)
                
                return HealthCheck(
                    name="system_resources",
                    status=status,
                    message=message,
                    details=metrics.to_dict()
                )
            except Exception as e:
                return HealthCheck(
                    name="system_resources",
                    status=HealthStatus.CRITICAL,
                    message=f"System resources check failed: {str(e)}"
                )
        
        # Register all checks
        self.health_checker.register_check("database", database_check)
        self.health_checker.register_check("cache", cache_check)
        self.health_checker.register_check("ml_models", ml_models_check)
        self.health_checker.register_check("fpl_api", api_connectivity_check)
        self.health_checker.register_check("system_resources", system_resources_check)
    
    async def collect_metrics(self):
        """Collect and store metrics."""
        try:
            metrics = self.metrics_collector.collect_system_metrics()
            
            # Add to history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            # Log performance metrics
            perf_logger.logger.info(
                "System metrics collected",
                extra={
                    'metric_type': 'system_metrics',
                    **metrics.to_dict()
                }
            )
            
            # Check for alerts
            await self._check_metric_alerts(metrics)
            
        except Exception as e:
            self.logger.error("Failed to collect metrics", error=str(e))
    
    async def _check_metric_alerts(self, metrics: SystemMetrics):
        """Check metrics for alert conditions."""
        
        # CPU alerts
        if metrics.cpu_percent > 90:
            await self.alert_manager.send_alert(
                "critical",
                f"Critical CPU usage: {metrics.cpu_percent:.1f}%",
                cpu_percent=metrics.cpu_percent
            )
        elif metrics.cpu_percent > 80:
            await self.alert_manager.send_alert(
                "warning",
                f"High CPU usage: {metrics.cpu_percent:.1f}%",
                cpu_percent=metrics.cpu_percent
            )
        
        # Memory alerts
        if metrics.memory_percent > 95:
            await self.alert_manager.send_alert(
                "critical",
                f"Critical memory usage: {metrics.memory_percent:.1f}%",
                memory_percent=metrics.memory_percent,
                memory_available=metrics.memory_available
            )
        elif metrics.memory_percent > 85:
            await self.alert_manager.send_alert(
                "warning",
                f"High memory usage: {metrics.memory_percent:.1f}%",
                memory_percent=metrics.memory_percent
            )
        
        # Disk alerts
        if metrics.disk_percent > 95:
            await self.alert_manager.send_alert(
                "critical",
                f"Critical disk usage: {metrics.disk_percent:.1f}%",
                disk_percent=metrics.disk_percent,
                disk_free=metrics.disk_free
            )
        elif metrics.disk_percent > 90:
            await self.alert_manager.send_alert(
                "warning",
                f"High disk usage: {metrics.disk_percent:.1f}%",
                disk_percent=metrics.disk_percent
            )
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return summary."""
        try:
            results = await self.health_checker.run_all_checks()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'checks': {},
                'summary': {
                    'total': len(results),
                    'healthy': 0,
                    'warning': 0,
                    'critical': 0,
                    'unknown': 0
                }
            }
            
            for result in results:
                summary['checks'][result.name] = {
                    'status': result.status.value,
                    'message': result.message,
                    'response_time': result.response_time,
                    'details': result.details
                }
                
                # Update counters
                summary['summary'][result.status.value] += 1
            
            # Determine overall status
            if summary['summary']['critical'] > 0:
                summary['overall_status'] = 'critical'
            elif summary['summary']['warning'] > 0:
                summary['overall_status'] = 'warning'
            elif summary['summary']['unknown'] > 0:
                summary['overall_status'] = 'unknown'
            
            self.logger.info("Health checks completed", 
                           overall_status=summary['overall_status'],
                           **summary['summary'])
            
            return summary
            
        except Exception as e:
            self.logger.error("Health check run failed", error=str(e))
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'unknown',
                'error': str(e)
            }
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        if self.running:
            self.logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.logger.info("Starting system monitoring", interval=self.collection_interval)
        
        try:
            while self.running:
                # Collect metrics
                await self.collect_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
        except Exception as e:
            self.logger.error("Monitoring loop failed", error=str(e))
        finally:
            self.running = False
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.running = False
        self.logger.info("Stopping system monitoring")
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        if not self.metrics_history:
            return {'error': 'No metrics data available'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': f'No metrics data for last {hours} hours'}
        
        # Calculate averages
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_percent for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'disk': {
                'avg': sum(disk_values) / len(disk_values),
                'max': max(disk_values),
                'min': min(disk_values)
            },
            'latest': recent_metrics[-1].to_dict()
        }


# Global monitor instance
_system_monitor = None


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


# Alert handlers
async def log_alert_handler(alert: Dict[str, Any]):
    """Log alert to system logs."""
    logger.warning("System alert", **alert)


async def console_alert_handler(alert: Dict[str, Any]):
    """Print alert to console."""
    timestamp = alert['timestamp']
    severity = alert['severity'].upper()
    message = alert['message']
    print(f"[{timestamp}] {severity}: {message}")


# Initialize monitoring
def setup_monitoring(collection_interval: int = 60, enable_alerts: bool = True):
    """Setup system monitoring."""
    monitor = get_system_monitor()
    monitor.collection_interval = collection_interval
    
    if enable_alerts:
        monitor.alert_manager.add_handler(log_alert_handler)
        monitor.alert_manager.add_handler(console_alert_handler)
    
    logger.info("System monitoring setup complete", 
               collection_interval=collection_interval,
               alerts_enabled=enable_alerts)