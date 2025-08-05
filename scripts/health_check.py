#!/usr/bin/env python3
"""
System health check script for FPL ML System.
Can be run standalone or as part of monitoring pipeline.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.monitoring import get_system_monitor
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def main():
    """Run comprehensive system health check."""
    
    print("🏥 FPL ML System Health Check")
    print("=" * 50)
    
    try:
        monitor = get_system_monitor()
        
        # Run health checks
        print("🔍 Running health checks...")
        health_results = await monitor.run_health_checks()
        
        # Collect system metrics
        print("📊 Collecting system metrics...")
        await monitor.collect_metrics()
        metrics_summary = monitor.get_metrics_summary(hours=1)
        
        # Display results
        print("\n📋 Health Check Results:")
        print("-" * 30)
        
        overall_status = health_results.get('overall_status', 'unknown')
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'critical': '❌',
            'unknown': '❓'
        }
        
        print(f"Overall Status: {status_emoji.get(overall_status, '❓')} {overall_status.upper()}")
        
        # Individual check results
        for check_name, check_result in health_results.get('checks', {}).items():
            status = check_result['status']
            message = check_result['message']
            response_time = check_result.get('response_time', 0)
            
            emoji = status_emoji.get(status, '❓')
            print(f"  {emoji} {check_name}: {message} ({response_time:.3f}s)")
        
        # Summary counts
        summary = health_results.get('summary', {})
        print(f"\nSummary: {summary.get('healthy', 0)} healthy, {summary.get('warning', 0)} warnings, {summary.get('critical', 0)} critical")
        
        # System metrics
        if metrics_summary and 'error' not in metrics_summary:
            print("\n📈 System Metrics (Last Hour):")
            print("-" * 30)
            
            cpu = metrics_summary.get('cpu', {})
            memory = metrics_summary.get('memory', {})
            disk = metrics_summary.get('disk', {})
            
            print(f"  CPU: {cpu.get('avg', 0):.1f}% avg, {cpu.get('max', 0):.1f}% max")
            print(f"  Memory: {memory.get('avg', 0):.1f}% avg, {memory.get('max', 0):.1f}% max")
            print(f"  Disk: {disk.get('avg', 0):.1f}% avg, {disk.get('max', 0):.1f}% max")
            
            print(f"  Data Points: {metrics_summary.get('data_points', 0)}")
        
        # Exit with appropriate code
        if overall_status == 'critical':
            print("\n🚨 CRITICAL issues detected!")
            sys.exit(2)
        elif overall_status == 'warning':
            print("\n⚠️ Warning conditions detected")
            sys.exit(1)
        else:
            print("\n✅ System is healthy!")
            sys.exit(0)
    
    except Exception as e:
        logger.exception("Health check failed")
        print(f"\n❌ Health check failed: {str(e)}")
        sys.exit(3)


def check_individual_component(component: str):
    """Check individual system component."""
    
    async def run_single_check():
        monitor = get_system_monitor()
        result = await monitor.health_checker.run_check(component)
        
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️', 
            'critical': '❌',
            'unknown': '❓'
        }
        
        emoji = status_emoji.get(result.status.value, '❓')
        print(f"{emoji} {component}: {result.message}")
        
        if result.details:
            print(f"   Details: {json.dumps(result.details, indent=2)}")
        
        if result.response_time:
            print(f"   Response Time: {result.response_time:.3f}s")
        
        # Exit with status code
        if result.status.value == 'critical':
            sys.exit(2)
        elif result.status.value == 'warning':
            sys.exit(1)
        else:
            sys.exit(0)
    
    asyncio.run(run_single_check())


def show_metrics():
    """Show current system metrics."""
    
    async def collect_and_show():
        monitor = get_system_monitor()
        await monitor.collect_metrics()
        
        if monitor.metrics_history:
            latest = monitor.metrics_history[-1]
            metrics_dict = latest.to_dict()
            print(json.dumps(metrics_dict, indent=2))
        else:
            print("No metrics data available")
    
    asyncio.run(collect_and_show())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FPL ML System Health Check")
    parser.add_argument("--component", help="Check specific component only")
    parser.add_argument("--metrics", action="store_true", help="Show current metrics")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    if args.component:
        check_individual_component(args.component)
    elif args.metrics:
        show_metrics()
    else:
        asyncio.run(main())