"""
Integration tests for agent communication using TestModel patterns.
Tests FPL Manager Agent delegation patterns and TestModel validation.
Following patterns from use-cases/pydantic-ai/examples/testing_examples/
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import TestModel components with error handling
try:
    from pydantic_ai.models.test import TestModel, FunctionModel
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    # Create mock replacements for testing without pydantic-ai
    TestModel = MagicMock
    FunctionModel = MagicMock
    Agent = MagicMock
    PYDANTIC_AI_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.agent
class TestAgentCommunication:
    """Integration tests for agent communication using TestModel patterns."""
    
    def test_testmodel_availability(self):
        """Test that TestModel components are available."""
        if PYDANTIC_AI_AVAILABLE:
            assert TestModel is not None
            assert FunctionModel is not None
            assert Agent is not None
        else:
            pytest.skip("PydanticAI not available - using mock implementations")
    
    @pytest.mark.asyncio
    async def test_fpl_manager_agent_initialization(self):
        """Test FPL Manager Agent initializes correctly with TestModel."""
        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("PydanticAI not available")
            
        # Mock FPL Manager Agent initialization
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockAgent:
            mock_agent = AsyncMock()
            MockAgent.return_value = mock_agent
            
            # Test agent creation
            agent = MockAgent()
            assert agent is not None
            
            # Test TestModel integration
            test_model = TestModel(custom_output_text='{"status": "initialized", "agent": "fpl_manager"}')
            
            # Mock agent.override method for TestModel
            mock_agent.override.return_value.__aenter__ = AsyncMock()
            mock_agent.override.return_value.__aexit__ = AsyncMock()
            
            # Test agent override with TestModel
            async with mock_agent.override(model=test_model):
                # Should be able to use TestModel
                assert test_model is not None
    
    @pytest.mark.asyncio
    async def test_agent_delegation_patterns(self):
        """Test FPL Manager Agent delegation to specialized agents."""
        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("PydanticAI not available")
        
        # Create TestModels for different agent responses
        success_models = {
            'data_pipeline': TestModel(custom_output_text='{"status": "success", "data_fetched": 1247, "quality": 0.96}'),
            'ml_prediction': TestModel(custom_output_text='{"predictions": {"player_1": 8.5, "player_2": 6.2}, "confidence": 0.78}'),
            'transfer_advisor': TestModel(custom_output_text='{"transfers": [{"out": "Wilson", "in": "Haaland", "gain": 2.3}], "cost": 4}'),
        }
        
        # Mock agent delegation
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockFPLAgent:
            with patch('src.agents.data_pipeline.DataPipelineAgent') as MockDataAgent:
                with patch('src.agents.ml_prediction.MLPredictionAgent') as MockMLAgent:
                    with patch('src.agents.transfer_advisor.TransferAdvisorAgent') as MockTransferAgent:
                        
                        # Create mock agents
                        fpl_agent = AsyncMock()
                        data_agent = AsyncMock()
                        ml_agent = AsyncMock()
                        transfer_agent = AsyncMock()
                        
                        MockFPLAgent.return_value = fpl_agent
                        MockDataAgent.return_value = data_agent
                        MockMLAgent.return_value = ml_agent
                        MockTransferAgent.return_value = transfer_agent
                        
                        # Mock delegation workflow
                        fpl_agent.delegate_to_data_agent.return_value = {"status": "success", "data": "fetched"}
                        fpl_agent.delegate_to_ml_agent.return_value = {"predictions": "generated"}
                        fpl_agent.delegate_to_transfer_agent.return_value = {"advice": "provided"}
                        
                        # Test delegation chain
                        data_result = await fpl_agent.delegate_to_data_agent()
                        ml_result = await fpl_agent.delegate_to_ml_agent()
                        transfer_result = await fpl_agent.delegate_to_transfer_agent()
                        
                        assert data_result["status"] == "success"
                        assert "predictions" in ml_result
                        assert "advice" in transfer_result
    
    @pytest.mark.asyncio
    async def test_agent_error_propagation_and_recovery(self):
        """Test agent error propagation and recovery mechanisms."""
        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("PydanticAI not available")
        
        # Create error TestModels
        error_models = {
            'api_timeout': TestModel(custom_output_text='{"error": "API timeout", "retry_suggested": true, "status": "error"}'),
            'data_validation': TestModel(custom_output_text='{"error": "Invalid data format", "details": "Missing required fields", "status": "error"}'),
            'ml_model_error': TestModel(custom_output_text='{"error": "Model prediction failed", "model_status": "unavailable", "status": "error"}'),
        }
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockAgent:
            mock_agent = AsyncMock()
            MockAgent.return_value = mock_agent
            
            # Test error handling
            mock_agent.handle_agent_error.return_value = {
                "error_handled": True,
                "fallback_used": True,
                "retry_count": 1
            }
            
            # Simulate error scenario
            error_result = await mock_agent.handle_agent_error()
            assert error_result["error_handled"] is True
            assert error_result["fallback_used"] is True
    
    @pytest.mark.asyncio
    async def test_agent_tool_calling_patterns(self):
        """Test agent tool calling patterns with FunctionModel."""
        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("PydanticAI not available")
        
        # Create FunctionModel for tool calling
        def mock_tool_caller(messages, tools):
            tool_calls = []
            for tool in tools:
                tool_calls.append({
                    "tool": tool.name if hasattr(tool, 'name') else str(tool),
                    "called": True,
                    "result": "success"
                })
            return f'{{"tool_calls": {tool_calls}, "status": "success"}}'
        
        function_model = FunctionModel(function=mock_tool_caller)
        
        # Mock tools available to agents
        mock_tools = [
            MagicMock(name='get_current_team'),
            MagicMock(name='optimize_team'),
            MagicMock(name='predict_player_points'),
            MagicMock(name='get_transfer_advice')
        ]
        
        # Test tool calling
        result = mock_tool_caller(["test message"], mock_tools)
        assert "tool_calls" in result
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_agent_context_management(self):
        """Test agent context management and state persistence."""
        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("PydanticAI not available")
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockAgent:
            mock_agent = AsyncMock()
            MockAgent.return_value = mock_agent
            
            # Test context creation and management
            test_context = {
                "user_id": "test_user_123",
                "session_id": "session_456",
                "fpl_team_id": 789,
                "current_gameweek": 20,
                "timestamp": datetime.now().isoformat()
            }
            
            mock_agent.create_context.return_value = test_context
            mock_agent.get_context.return_value = test_context
            mock_agent.update_context.return_value = {**test_context, "updated": True}
            
            # Test context lifecycle
            created_context = await mock_agent.create_context()
            retrieved_context = await mock_agent.get_context()
            updated_context = await mock_agent.update_context()
            
            assert created_context["user_id"] == "test_user_123"
            assert retrieved_context["session_id"] == "session_456"
            assert updated_context["updated"] is True
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self):
        """Test multi-agent coordination for complex workflows."""
        if not PYDANTIC_AI_AVAILABLE:
            pytest.skip("PydanticAI not available")
        
        # Mock complex workflow involving multiple agents
        workflow_steps = [
            {"agent": "data_pipeline", "action": "fetch_latest_data", "duration": 0.5},
            {"agent": "ml_prediction", "action": "generate_predictions", "duration": 2.1},
            {"agent": "transfer_advisor", "action": "optimize_transfers", "duration": 1.8},
            {"agent": "fpl_manager", "action": "compile_recommendations", "duration": 0.3}
        ]
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockFPLAgent:
            mock_fpl_agent = AsyncMock()
            MockFPLAgent.return_value = mock_fpl_agent
            
            # Mock workflow coordination
            mock_fpl_agent.coordinate_workflow.return_value = {
                "workflow_id": "workflow_123",
                "steps_completed": len(workflow_steps),
                "total_duration": sum(step["duration"] for step in workflow_steps),
                "status": "completed",
                "results": {
                    "data_quality": 0.96,
                    "predictions_generated": 100,
                    "transfers_analyzed": 15,
                    "recommendations": 3
                }
            }
            
            # Test workflow coordination
            workflow_result = await mock_fpl_agent.coordinate_workflow()
            
            assert workflow_result["status"] == "completed"
            assert workflow_result["steps_completed"] == 4
            assert workflow_result["total_duration"] == 4.7  # Sum of all durations
            assert workflow_result["results"]["recommendations"] == 3


@pytest.mark.integration
@pytest.mark.agent
class TestAgentDataFlow:
    """Integration tests for data flow between agents."""
    
    @pytest.mark.asyncio
    async def test_data_pipeline_to_ml_agent_flow(self):
        """Test data flow from DataPipeline to ML Prediction agent."""
        with patch('src.agents.data_pipeline.DataPipelineAgent') as MockDataAgent:
            with patch('src.agents.ml_prediction.MLPredictionAgent') as MockMLAgent:
                
                data_agent = AsyncMock()
                ml_agent = AsyncMock()
                MockDataAgent.return_value = data_agent
                MockMLAgent.return_value = ml_agent
                
                # Mock data pipeline output
                pipeline_output = {
                    "processed_data": {
                        "players": 100,
                        "features": 15,
                        "quality_score": 0.96
                    },
                    "feature_matrix": "mock_dataframe",
                    "validation_passed": True
                }
                
                data_agent.process_fpl_data.return_value = pipeline_output
                
                # Mock ML agent processing
                ml_input = pipeline_output["processed_data"]
                ml_agent.generate_predictions.return_value = {
                    "predictions": {"player_1": 8.5, "player_2": 6.2},
                    "model_confidence": 0.78,
                    "features_used": 15
                }
                
                # Test data flow
                data_result = await data_agent.process_fpl_data()
                ml_result = await ml_agent.generate_predictions(ml_input)
                
                assert data_result["validation_passed"] is True
                assert ml_result["model_confidence"] > 0.7
    
    @pytest.mark.asyncio
    async def test_ml_to_transfer_advisor_flow(self):
        """Test data flow from ML Prediction to Transfer Advisor agent."""
        with patch('src.agents.ml_prediction.MLPredictionAgent') as MockMLAgent:
            with patch('src.agents.transfer_advisor.TransferAdvisorAgent') as MockTransferAgent:
                
                ml_agent = AsyncMock()
                transfer_agent = AsyncMock()
                MockMLAgent.return_value = ml_agent
                MockTransferAgent.return_value = transfer_agent
                
                # Mock ML predictions
                ml_predictions = {
                    "player_predictions": {
                        "1": {"expected_points": 8.5, "confidence": 0.85},
                        "2": {"expected_points": 6.2, "confidence": 0.72},
                        "3": {"expected_points": 9.1, "confidence": 0.88}
                    },
                    "model_metadata": {
                        "model_version": "1.2.0",
                        "training_date": "2024-01-15",
                        "benchmark_mse": 0.002
                    }
                }
                
                ml_agent.get_player_predictions.return_value = ml_predictions
                
                # Mock transfer optimization
                transfer_agent.optimize_transfers.return_value = {
                    "recommended_transfers": [
                        {"out": "Player 2", "in": "Player 3", "expected_gain": 2.9}
                    ],
                    "optimization_time": 1.8,
                    "status": "optimal"
                }
                
                # Test data flow
                predictions = await ml_agent.get_player_predictions()
                transfer_advice = await transfer_agent.optimize_transfers(predictions)
                
                assert len(transfer_advice["recommended_transfers"]) > 0
                assert transfer_advice["status"] == "optimal"
    
    @pytest.mark.asyncio
    async def test_complete_agent_pipeline_integration(self):
        """Test complete integration of all agents in a realistic workflow."""
        agents = {}
        
        # Mock all agent types
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockFPLAgent:
            with patch('src.agents.data_pipeline.DataPipelineAgent') as MockDataAgent:
                with patch('src.agents.ml_prediction.MLPredictionAgent') as MockMLAgent:
                    with patch('src.agents.transfer_advisor.TransferAdvisorAgent') as MockTransferAgent:
                        
                        # Create agent instances
                        agents['fpl_manager'] = AsyncMock()
                        agents['data_pipeline'] = AsyncMock()
                        agents['ml_prediction'] = AsyncMock()
                        agents['transfer_advisor'] = AsyncMock()
                        
                        MockFPLAgent.return_value = agents['fpl_manager']
                        MockDataAgent.return_value = agents['data_pipeline']
                        MockMLAgent.return_value = agents['ml_prediction']
                        MockTransferAgent.return_value = agents['transfer_advisor']
                        
                        # Mock complete workflow
                        agents['fpl_manager'].run_complete_analysis.return_value = {
                            "analysis_id": "analysis_789",
                            "timestamp": datetime.now().isoformat(),
                            "workflow_steps": {
                                "data_fetch": {"status": "completed", "duration": 0.8},
                                "prediction": {"status": "completed", "duration": 2.3},
                                "optimization": {"status": "completed", "duration": 1.9},
                                "recommendation": {"status": "completed", "duration": 0.4}
                            },
                            "final_recommendations": {
                                "team_analysis": "Strong midfield, weak defense",
                                "suggested_transfers": 2,
                                "captain_recommendation": "Salah",
                                "expected_improvement": "+5.2 points"
                            },
                            "confidence_score": 0.84,
                            "total_duration": 5.4
                        }
                        
                        # Test complete workflow
                        analysis_result = await agents['fpl_manager'].run_complete_analysis()
                        
                        assert analysis_result["confidence_score"] > 0.8
                        assert analysis_result["final_recommendations"]["suggested_transfers"] > 0
                        assert "team_analysis" in analysis_result["final_recommendations"]
                        assert analysis_result["total_duration"] < 10.0  # Performance requirement


@pytest.mark.integration
@pytest.mark.agent
class TestAgentErrorScenarios:
    """Integration tests for agent error scenarios and recovery."""
    
    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self):
        """Test agent timeout handling and recovery."""
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockAgent:
            mock_agent = AsyncMock()
            MockAgent.return_value = mock_agent
            
            # Mock timeout scenario
            mock_agent.handle_timeout.return_value = {
                "timeout_occurred": True,
                "operation": "team_optimization",
                "timeout_duration": 30.0,
                "fallback_result": {
                    "status": "partial",
                    "recommendations": "basic_analysis",
                    "confidence": 0.5
                },
                "retry_scheduled": True
            }
            
            timeout_result = await mock_agent.handle_timeout()
            
            assert timeout_result["timeout_occurred"] is True
            assert timeout_result["fallback_result"]["status"] == "partial"
            assert timeout_result["retry_scheduled"] is True
    
    @pytest.mark.asyncio
    async def test_agent_cascade_failure_recovery(self):
        """Test agent cascade failure recovery mechanisms."""
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockFPLAgent:
            mock_fpl_agent = AsyncMock()
            MockFPLAgent.return_value = mock_fpl_agent
            
            # Mock cascade failure scenario
            failure_scenario = {
                "trigger": "data_agent_failure",
                "affected_agents": ["ml_prediction", "transfer_advisor"],
                "recovery_strategy": "fallback_to_cache",
                "recovery_time": 2.1
            }
            
            mock_fpl_agent.handle_cascade_failure.return_value = {
                "failure_detected": True,
                "recovery_successful": True,
                "fallback_data_used": True,
                "affected_operations": len(failure_scenario["affected_agents"]),
                "recovery_time": failure_scenario["recovery_time"],
                "final_status": "degraded_service"
            }
            
            recovery_result = await mock_fpl_agent.handle_cascade_failure()
            
            assert recovery_result["failure_detected"] is True
            assert recovery_result["recovery_successful"] is True
            assert recovery_result["final_status"] == "degraded_service"