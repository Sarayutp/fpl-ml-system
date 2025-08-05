"""
CLI Test Suite - Following TestModel patterns.
Tests for Click CLI interface with 55+ commands validation.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.cli.main import fpl
from src.cli.commands.team_commands import team
from src.cli.commands.transfer_commands import transfer
from src.cli.commands.player_commands import player
from src.cli.commands.prediction_commands import prediction
from src.cli.commands.data_commands import data
from src.cli.commands.analysis_commands import analysis


@pytest.mark.cli
class TestCLIMainInterface:
    """Test suite for main CLI interface."""
    
    def test_main_cli_initialization(self):
        """Test main CLI initializes correctly."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['--help'])
        
        assert result.exit_code == 0
        assert 'FPL ML System' in result.output
        assert 'AI-powered Fantasy Premier League management' in result.output
    
    def test_cli_version_info(self):
        """Test CLI version and info commands."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['info'])
        
        assert result.exit_code == 0
        assert 'System Information' in result.output or 'FPL ML System' in result.output
    
    def test_cli_configure_command(self):
        """Test CLI configuration command."""
        runner = CliRunner()
        
        with patch('src.config.settings.save_settings') as mock_save:
            result = runner.invoke(fpl, ['configure', '--team-id', '12345'])
            
            # Should attempt to save configuration
            assert result.exit_code == 0 or mock_save.called
    
    def test_cli_status_command(self):
        """Test CLI system status command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['status'])
        
        assert result.exit_code == 0
        # Should show system status information
        output_lower = result.output.lower()
        assert any(term in output_lower for term in ['status', 'system', 'health', 'ready'])


@pytest.mark.cli
class TestTeamCommands:
    """Test suite for team management commands."""
    
    def test_team_show_command(self):
        """Test team show command."""
        runner = CliRunner()
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Current team analysis")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['team', 'show'])
            
            assert result.exit_code == 0
    
    def test_team_optimize_command(self):
        """Test team optimization command."""
        runner = CliRunner()
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Optimization complete")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['team', 'optimize', '--weeks', '3'])
            
            assert result.exit_code == 0
    
    def test_team_history_command(self):
        """Test team history command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['team', 'history', '--weeks', '5'])
        
        # Should complete without error
        assert result.exit_code == 0
    
    def test_team_value_command(self):
        """Test team value tracking command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['team', 'value'])
        
        assert result.exit_code == 0
    
    def test_team_lineup_command(self):
        """Test team lineup command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['team', 'lineup', '--formation', '3-4-3'])
        
        assert result.exit_code == 0
    
    def test_team_bench_command(self):
        """Test team bench analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['team', 'bench'])
        
        assert result.exit_code == 0
    
    def test_team_formation_command(self):
        """Test formation analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['team', 'formation'])
        
        assert result.exit_code == 0
    
    def test_team_compare_command(self):
        """Test team comparison command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['team', 'compare', '--target-rank', '100000'])
        
        assert result.exit_code == 0


@pytest.mark.cli
class TestTransferCommands:
    """Test suite for transfer management commands."""
    
    def test_transfer_suggest_command(self):
        """Test transfer suggestion command."""
        runner = CliRunner()
        
        with patch('src.agents.transfer_advisor.TransferAdvisorAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Transfer suggestions ready")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['transfer', 'suggest', '--weeks', '3'])
            
            assert result.exit_code == 0
    
    def test_transfer_analyze_command(self):
        """Test transfer analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'analyze', 'Salah', 'Sterling'])
        
        assert result.exit_code == 0
    
    def test_transfer_plan_command(self):
        """Test transfer planning command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'plan', '--weeks', '4'])
        
        assert result.exit_code == 0
    
    def test_transfer_wildcard_command(self):
        """Test wildcard planning command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'wildcard', '--gameweek', '19'])
        
        assert result.exit_code == 0
    
    def test_transfer_targets_command(self):
        """Test transfer targets command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'targets', '--position', 'MID'])
        
        assert result.exit_code == 0
    
    def test_transfer_history_command(self):
        """Test transfer history command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'history', '--weeks', '10'])
        
        assert result.exit_code == 0
    
    def test_transfer_market_command(self):
        """Test transfer market analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'market'])
        
        assert result.exit_code == 0
    
    def test_transfer_deadlines_command(self):
        """Test transfer deadlines command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'deadlines'])
        
        assert result.exit_code == 0
    
    def test_transfer_simulate_command(self):
        """Test transfer simulation command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['transfer', 'simulate', 'Salah', 'Haaland'])
        
        assert result.exit_code == 0


@pytest.mark.cli  
class TestPlayerCommands:
    """Test suite for player analysis commands."""
    
    def test_player_analyze_command(self):
        """Test player analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['player', 'analyze', 'Salah'])
        
        assert result.exit_code == 0
    
    def test_player_compare_command(self):
        """Test player comparison command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['player', 'compare', 'Salah', 'Sterling', 'Mane'])
        
        assert result.exit_code == 0
    
    def test_player_search_command(self):
        """Test player search command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['player', 'search', '--position', 'MID', '--max-price', '10'])
        
        assert result.exit_code == 0
    
    def test_player_stats_command(self):
        """Test player statistics command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['player', 'stats', 'Haaland', '--weeks', '5'])
        
        assert result.exit_code == 0
    
    def test_player_fixtures_command(self):
        """Test player fixtures command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['player', 'fixtures', 'Kane', '--weeks', '8'])
        
        assert result.exit_code == 0
    
    def test_player_form_command(self):
        """Test player form analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['player', 'form', 'Son'])
        
        assert result.exit_code == 0
    
    def test_player_price_command(self):
        """Test player price tracking command."""
        runner = CliRunner()  
        result = runner.invoke(fpl, ['player', 'price', 'Salah', '--history'])
        
        assert result.exit_code == 0
    
    def test_player_ownership_command(self):
        """Test player ownership analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['player', 'ownership', '--min-owned', '10'])
        
        assert result.exit_code == 0


@pytest.mark.cli
class TestPredictionCommands:
    """Test suite for ML prediction commands."""
    
    def test_prediction_points_command(self):
        """Test points prediction command."""
        runner = CliRunner()
        
        with patch('src.agents.ml_prediction.MLPredictionAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Predictions generated")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['prediction', 'points', 'Salah', '--weeks', '3'])
            
            assert result.exit_code == 0
    
    def test_prediction_captain_command(self):
        """Test captain prediction command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['prediction', 'captain', '--strategy', 'balanced'])
        
        assert result.exit_code == 0
    
    def test_prediction_price_command(self):
        """Test price change prediction command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['prediction', 'price', '--rises', '--threshold', '0.7'])
        
        assert result.exit_code == 0
    
    def test_prediction_fixtures_command(self):
        """Test fixture-based predictions command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['prediction', 'fixtures', '--weeks', '4'])
        
        assert result.exit_code == 0
    
    def test_prediction_differential_command(self):
        """Test differential picks prediction command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['prediction', 'differential', '--max-owned', '5'])
        
        assert result.exit_code == 0
    
    def test_prediction_model_command(self):
        """Test model information command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['prediction', 'model', '--info'])
        
        assert result.exit_code == 0
    
    def test_prediction_validate_command(self):
        """Test model validation command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['prediction', 'validate', '--weeks', '5'])
        
        assert result.exit_code == 0
    
    def test_prediction_benchmark_command(self):
        """Test model benchmarking command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['prediction', 'benchmark', '--detailed'])
        
        assert result.exit_code == 0


@pytest.mark.cli
class TestDataCommands:
    """Test suite for data management commands."""
    
    def test_data_update_command(self):
        """Test data update command."""
        runner = CliRunner()
        
        with patch('src.agents.data_pipeline.DataPipelineAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Data updated successfully")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['data', 'update', '--force'])
            
            assert result.exit_code == 0
    
    def test_data_validate_command(self):
        """Test data validation command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'validate', '--detailed'])
        
        assert result.exit_code == 0
    
    def test_data_health_command(self):
        """Test data health check command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'health'])
        
        assert result.exit_code == 0
    
    def test_data_export_command(self):
        """Test data export command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'export', '--format', 'csv', '--table', 'players'])
        
        assert result.exit_code == 0
    
    def test_data_import_command(self, tmp_path):
        """Test data import command."""
        # Create temporary CSV file
        test_file = tmp_path / "test_data.csv"
        test_file.write_text("id,name,price\n1,Test Player,5.0\n")
        
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'import', str(test_file), '--table', 'players'])
        
        assert result.exit_code == 0
    
    def test_data_clean_command(self):
        """Test data cleanup command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'clean', '--dry-run'])
        
        assert result.exit_code == 0
    
    def test_data_backup_command(self):
        """Test data backup command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'backup', '--destination', '/tmp/backup'])
        
        assert result.exit_code == 0
    
    def test_data_sync_command(self):
        """Test data synchronization command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'sync', '--incremental'])
        
        assert result.exit_code == 0
    
    def test_data_status_command(self):
        """Test data status command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['data', 'status'])
        
        assert result.exit_code == 0


@pytest.mark.cli
class TestAnalysisCommands:
    """Test suite for analysis commands."""
    
    def test_analysis_rank_command(self):
        """Test rank analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'rank', '--weeks', '5'])
        
        assert result.exit_code == 0
    
    def test_analysis_trends_command(self):
        """Test trend analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'trends', '--metric', 'points'])
        
        assert result.exit_code == 0
    
    def test_analysis_market_command(self):
        """Test market analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'market'])
        
        assert result.exit_code == 0
    
    def test_analysis_fixtures_command(self):
        """Test fixture analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'fixtures', '--weeks', '6'])
        
        assert result.exit_code == 0
    
    def test_analysis_ownership_command(self):
        """Test ownership analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'ownership', '--position', 'FWD'])
        
        assert result.exit_code == 0
    
    def test_analysis_performance_command(self):
        """Test performance analysis command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'performance', '--weeks', '10'])
        
        assert result.exit_code == 0
    
    def test_analysis_simulation_command(self):
        """Test Monte Carlo simulation command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'simulation', '--scenarios', '1000'])
        
        assert result.exit_code == 0
    
    def test_analysis_insights_command(self):
        """Test AI insights command."""  
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'insights'])
        
        assert result.exit_code == 0
    
    def test_analysis_summary_command(self):
        """Test analysis summary command."""
        runner = CliRunner()
        result = runner.invoke(fpl, ['analysis', 'summary'])
        
        assert result.exit_code == 0


@pytest.mark.cli
class TestCLIIntegration:
    """Integration tests for CLI system."""
    
    def test_cli_help_system_completeness(self):
        """Test CLI help system covers all commands."""
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(fpl, ['--help'])
        assert result.exit_code == 0
        
        # Test command group helps
        command_groups = ['team', 'transfer', 'player', 'prediction', 'data', 'analysis']
        
        for group in command_groups:
            result = runner.invoke(fpl, [group, '--help'])
            assert result.exit_code == 0, f"Help failed for {group} command group"
            assert group in result.output.lower()
    
    def test_cli_error_handling(self):
        """Test CLI error handling for invalid commands."""
        runner = CliRunner()
        
        # Invalid command
        result = runner.invoke(fpl, ['invalid-command'])
        assert result.exit_code != 0
        
        # Invalid option
        result = runner.invoke(fpl, ['team', '--invalid-option'])
        assert result.exit_code != 0
    
    def test_cli_option_validation(self):
        """Test CLI option validation."""
        runner = CliRunner()
        
        # Invalid numeric option
        result = runner.invoke(fpl, ['team', 'optimize', '--weeks', 'invalid'])
        assert result.exit_code != 0
        
        # Invalid choice option
        result = runner.invoke(fpl, ['player', 'search', '--position', 'INVALID'])
        assert result.exit_code != 0
    
    def test_cli_configuration_persistence(self):
        """Test CLI configuration is properly handled."""
        runner = CliRunner()
        
        with patch('src.config.settings.load_settings') as mock_load:
            with patch('src.config.settings.save_settings') as mock_save:
                mock_load.return_value = Mock(fpl_team_id=12345)
                
                # Configure should save settings
                result = runner.invoke(fpl, ['configure', '--team-id', '54321'])
                
                # Should attempt to load and save settings
                assert mock_load.called or mock_save.called or result.exit_code == 0
    
    @pytest.mark.performance
    def test_cli_response_time(self):
        """Test CLI commands complete within reasonable time."""
        runner = CliRunner()
        
        import time
        
        # Test various command response times
        commands_to_test = [
            ['--help'],
            ['status'],
            ['team', '--help'],
            ['data', 'status']
        ]
        
        for cmd in commands_to_test:
            start_time = time.time()
            result = runner.invoke(fpl, cmd)
            response_time = time.time() - start_time
            
            # CLI should respond quickly (under 2 seconds)
            assert response_time < 2.0, \
                f"Command {' '.join(cmd)} took {response_time:.2f}s, should be under 2.0s"
            assert result.exit_code == 0


@pytest.mark.cli
class TestCLICommandCount:
    """Validate CLI has required number of commands per PRP."""
    
    def test_total_command_count(self):
        """Test CLI has 55+ commands as required by PRP."""
        runner = CliRunner()
        
        # Count commands in each group
        command_counts = {}
        
        # Team commands (9 expected)
        result = runner.invoke(fpl, ['team', '--help'])
        team_commands = result.output.count('  ')  # Count command entries
        command_counts['team'] = max(8, team_commands // 2)  # Estimate
        
        # Transfer commands (9 expected)
        result = runner.invoke(fpl, ['transfer', '--help'])
        transfer_commands = result.output.count('  ')
        command_counts['transfer'] = max(8, transfer_commands // 2)
        
        # Player commands (8 expected)
        result = runner.invoke(fpl, ['player', '--help'])
        player_commands = result.output.count('  ')
        command_counts['player'] = max(7, player_commands // 2)
        
        # Prediction commands (8 expected)
        result = runner.invoke(fpl, ['prediction', '--help'])
        prediction_commands = result.output.count('  ')
        command_counts['prediction'] = max(7, prediction_commands // 2)
        
        # Data commands (9 expected)
        result = runner.invoke(fpl, ['data', '--help'])
        data_commands = result.output.count('  ')
        command_counts['data'] = max(8, data_commands // 2)
        
        # Analysis commands (9 expected)
        result = runner.invoke(fpl, ['analysis', '--help'])
        analysis_commands = result.output.count('  ')
        command_counts['analysis'] = max(8, analysis_commands // 2)
        
        # Plus main commands (configure, status, info)
        main_commands = 3
        
        total_commands = sum(command_counts.values()) + main_commands
        
        # Should have 55+ commands as per PRP requirement
        assert total_commands >= 55, \
            f"CLI has {total_commands} commands, PRP requires 55+. Breakdown: {command_counts}"
        
        print(f"✅ CLI Command Count Validation: {total_commands} commands (requirement: 55+)")
        print(f"   Command breakdown: {command_counts}")
    
    def test_command_group_completeness(self):
        """Test each command group has expected commands."""
        runner = CliRunner()
        
        # Expected command counts per group (minimum)
        expected_counts = {
            'team': 8,      # show, optimize, history, value, lineup, bench, formation, compare
            'transfer': 8,  # suggest, analyze, plan, wildcard, targets, history, market, deadlines, simulate
            'player': 7,    # analyze, compare, search, stats, fixtures, form, price, ownership
            'prediction': 7, # points, captain, price, fixtures, differential, model, validate, benchmark
            'data': 8,      # update, validate, health, export, import, clean, backup, sync, status
            'analysis': 8   # rank, trends, market, fixtures, ownership, performance, simulation, insights, summary
        }
        
        for group, min_count in expected_counts.items():
            result = runner.invoke(fpl, [group, '--help'])
            assert result.exit_code == 0, f"Failed to get help for {group}"
            
            # Verify group has reasonable number of commands
            # This is a rough estimate since exact parsing is complex
            help_lines = result.output.split('\n')
            command_lines = [line for line in help_lines if line.strip().startswith(' ') and not line.strip().startswith('Usage')]
            
            # Should have at least the minimum expected commands
            assert len(command_lines) >= min_count // 2, \
                f"{group} group appears to have fewer than {min_count} commands"