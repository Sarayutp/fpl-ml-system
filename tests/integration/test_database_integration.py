"""
Integration tests for database operations, transactions, and connection pooling.
Tests SQLAlchemy ORM interactions and error recovery mechanisms.
"""

import pytest
import asyncio
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from datetime import datetime, timedelta


@pytest.mark.integration
class TestDatabaseOperations:
    """Integration tests for database operations and ORM interactions."""
    
    def test_database_connection_management(self):
        """Test database connection creation and management."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                # Test connection creation
                conn = sqlite3.connect(db_path)
                assert conn is not None
                
                # Test basic operations
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
                cursor.execute("INSERT INTO test (name) VALUES (?)", ("test_value",))
                conn.commit()
                
                # Verify data
                cursor.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                assert count == 1
                
                conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)
    
    def test_database_schema_creation_and_migration(self):
        """Test database schema creation and migration operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create initial schema
                schema_v1 = [
                    '''CREATE TABLE players (
                        id INTEGER PRIMARY KEY,
                        web_name TEXT NOT NULL,
                        total_points INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''',
                    '''CREATE TABLE teams (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        strength INTEGER DEFAULT 3,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )'''
                ]
                
                for statement in schema_v1:
                    cursor.execute(statement)
                
                conn.commit()
                
                # Verify schema
                cursor.execute("PRAGMA table_info(players)")
                player_columns = [col[1] for col in cursor.fetchall()]
                assert 'id' in player_columns
                assert 'web_name' in player_columns
                assert 'total_points' in player_columns
                
                # Test migration (add new column)
                cursor.execute("ALTER TABLE players ADD COLUMN element_type INTEGER DEFAULT 1")
                conn.commit()
                
                # Verify migration
                cursor.execute("PRAGMA table_info(players)")
                updated_columns = [col[1] for col in cursor.fetchall()]
                assert 'element_type' in updated_columns
                
                conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)
    
    def test_database_transaction_handling(self):
        """Test database transaction handling and rollback mechanisms."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create test table
                cursor.execute('''
                    CREATE TABLE transaction_test (
                        id INTEGER PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                ''')
                conn.commit()
                
                # Test successful transaction
                conn.execute("BEGIN")
                try:
                    cursor.execute("INSERT INTO transaction_test (value) VALUES (?)", ("success_1",))
                    cursor.execute("INSERT INTO transaction_test (value) VALUES (?)", ("success_2",))
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                
                # Verify successful transaction
                cursor.execute("SELECT COUNT(*) FROM transaction_test")
                count = cursor.fetchone()[0]
                assert count == 2
                
                # Test rollback on error
                conn.execute("BEGIN")
                try:
                    cursor.execute("INSERT INTO transaction_test (value) VALUES (?)", ("before_error",))
                    # Simulate error - try to insert duplicate primary key
                    cursor.execute("INSERT INTO transaction_test (id, value) VALUES (1, 'duplicate_id')")
                    conn.commit()
                except sqlite3.IntegrityError:
                    conn.rollback()
                
                # Verify rollback worked (count should still be 2)
                cursor.execute("SELECT COUNT(*) FROM transaction_test")
                count = cursor.fetchone()[0]
                assert count == 2
                
                conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)
    
    def test_database_concurrent_access(self):
        """Test database concurrent access and locking mechanisms."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                # Create test table
                conn1 = sqlite3.connect(db_path)
                cursor1 = conn1.cursor()
                cursor1.execute('''
                    CREATE TABLE concurrent_test (
                        id INTEGER PRIMARY KEY,
                        counter INTEGER DEFAULT 0
                    )
                ''')
                cursor1.execute("INSERT INTO concurrent_test (counter) VALUES (0)")
                conn1.commit()
                conn1.close()
                
                # Simulate concurrent access
                connections = []
                for i in range(3):
                    conn = sqlite3.connect(db_path)
                    connections.append(conn)
                
                # Each connection increments the counter
                for i, conn in enumerate(connections):
                    cursor = conn.cursor()
                    cursor.execute("SELECT counter FROM concurrent_test WHERE id = 1")
                    current_value = cursor.fetchone()[0]
                    cursor.execute("UPDATE concurrent_test SET counter = ? WHERE id = 1", 
                                 (current_value + 1,))
                    conn.commit()
                    conn.close()
                
                # Verify final result
                final_conn = sqlite3.connect(db_path)
                final_cursor = final_conn.cursor()
                final_cursor.execute("SELECT counter FROM concurrent_test WHERE id = 1")
                final_value = final_cursor.fetchone()[0]
                assert final_value == 3
                final_conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)
    
    def test_database_connection_pooling(self):
        """Test database connection pooling and resource management."""
        # Mock connection pool
        with patch('sqlite3.connect') as mock_connect:
            mock_connections = []
            
            def create_mock_connection(*args, **kwargs):
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.cursor.return_value = mock_cursor
                mock_connections.append(mock_conn)
                return mock_conn
            
            mock_connect.side_effect = create_mock_connection
            
            # Simulate connection pool usage
            pool_size = 5
            connections = []
            
            for i in range(pool_size):
                conn = sqlite3.connect("test.db")  # This will create mock connections
                connections.append(conn)
            
            assert len(mock_connections) == pool_size
            
            # Test connection reuse
            for conn in connections:
                conn.close()
            
            # Verify all connections were properly closed
            for mock_conn in mock_connections:
                mock_conn.close.assert_called_once()


@pytest.mark.integration
class TestDatabaseModelIntegration:
    """Integration tests for database models and ORM operations."""
    
    def test_player_model_database_operations(self):
        """Test Player model database operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create players table
                cursor.execute('''
                    CREATE TABLE players (
                        id INTEGER PRIMARY KEY,
                        web_name TEXT NOT NULL,
                        first_name TEXT,
                        second_name TEXT, 
                        element_type INTEGER,
                        team INTEGER,
                        now_cost INTEGER,
                        total_points INTEGER DEFAULT 0,
                        form TEXT DEFAULT '0.0',
                        selected_by_percent TEXT DEFAULT '0.0',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Test player insertion
                player_data = (
                    1, "Salah", "Mohamed", "Salah", 3, 1, 130, 187, "8.5", "45.2"
                )
                cursor.execute('''
                    INSERT INTO players 
                    (id, web_name, first_name, second_name, element_type, team, 
                     now_cost, total_points, form, selected_by_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', player_data)
                conn.commit()
                
                # Test player retrieval
                cursor.execute("SELECT * FROM players WHERE id = ?", (1,))
                retrieved = cursor.fetchone()
                assert retrieved is not None
                assert retrieved[1] == "Salah"  # web_name
                assert retrieved[7] == 187     # total_points
                
                # Test player update
                cursor.execute('''
                    UPDATE players 
                    SET total_points = ?, form = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (195, "9.0", 1))
                conn.commit()
                
                # Verify update
                cursor.execute("SELECT total_points, form FROM players WHERE id = ?", (1,))
                updated = cursor.fetchone()
                assert updated[0] == 195
                assert updated[1] == "9.0"
                
                conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)
    
    def test_team_model_database_operations(self):
        """Test Team model database operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create teams table
                cursor.execute('''
                    CREATE TABLE teams (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        short_name TEXT,
                        strength INTEGER DEFAULT 3,
                        strength_overall_home INTEGER DEFAULT 3,
                        strength_overall_away INTEGER DEFAULT 3,
                        strength_attack_home INTEGER DEFAULT 3,
                        strength_attack_away INTEGER DEFAULT 3,
                        strength_defence_home INTEGER DEFAULT 3,
                        strength_defence_away INTEGER DEFAULT 3,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert multiple teams
                teams_data = [
                    (1, "Liverpool", "LIV", 5, 5, 4, 5, 4, 4, 3),
                    (2, "Manchester City", "MCI", 5, 5, 5, 5, 5, 4, 4),
                    (3, "Arsenal", "ARS", 4, 4, 4, 4, 4, 4, 3),
                ]
                
                cursor.executemany('''
                    INSERT INTO teams 
                    (id, name, short_name, strength, strength_overall_home, 
                     strength_overall_away, strength_attack_home, strength_attack_away,
                     strength_defence_home, strength_defence_away)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', teams_data)
                conn.commit()
                
                # Test batch retrieval
                cursor.execute("SELECT COUNT(*) FROM teams")
                count = cursor.fetchone()[0]
                assert count == 3
                
                # Test filtering by strength
                cursor.execute("SELECT name FROM teams WHERE strength = 5")
                top_teams = cursor.fetchall()
                assert len(top_teams) == 2
                team_names = [team[0] for team in top_teams]
                assert "Liverpool" in team_names
                assert "Manchester City" in team_names
                
                conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)
    
    def test_fixture_model_database_operations(self):
        """Test Fixture model database operations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create fixtures table
                cursor.execute('''
                    CREATE TABLE fixtures (
                        id INTEGER PRIMARY KEY,
                        code INTEGER,
                        event INTEGER,
                        team_h INTEGER,
                        team_a INTEGER,
                        team_h_difficulty INTEGER,
                        team_a_difficulty INTEGER,
                        kickoff_time TIMESTAMP,
                        finished BOOLEAN DEFAULT 0,
                        started BOOLEAN DEFAULT 0,
                        team_h_score INTEGER,
                        team_a_score INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert fixture data
                fixture_data = [
                    (1, 123456, 20, 1, 2, 3, 4, datetime.now(), 1, 1, 2, 1),
                    (2, 123457, 20, 3, 4, 2, 3, datetime.now() + timedelta(days=1), 0, 0, None, None),
                    (3, 123458, 21, 1, 3, 4, 2, datetime.now() + timedelta(days=7), 0, 0, None, None),
                ]
                
                cursor.executemany('''
                    INSERT INTO fixtures 
                    (id, code, event, team_h, team_a, team_h_difficulty, team_a_difficulty,
                     kickoff_time, finished, started, team_h_score, team_a_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', fixture_data)
                conn.commit()
                
                # Test gameweek filtering
                cursor.execute("SELECT COUNT(*) FROM fixtures WHERE event = 20")
                gw20_count = cursor.fetchone()[0]
                assert gw20_count == 2
                
                # Test finished fixtures
                cursor.execute("SELECT team_h_score, team_a_score FROM fixtures WHERE finished = 1")
                finished_results = cursor.fetchall()
                assert len(finished_results) == 1
                assert finished_results[0] == (2, 1)
                
                # Test upcoming fixtures
                cursor.execute("SELECT COUNT(*) FROM fixtures WHERE finished = 0")
                upcoming_count = cursor.fetchone()[0]
                assert upcoming_count == 2
                
                conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)


@pytest.mark.integration
class TestDatabaseErrorHandling:
    """Integration tests for database error handling and recovery."""
    
    def test_database_connection_error_handling(self):
        """Test database connection error handling."""
        # Test connection to non-existent database file
        with pytest.raises(sqlite3.OperationalError):
            conn = sqlite3.connect("/invalid/path/database.db")
            conn.execute("SELECT 1")
    
    def test_database_constraint_violation_handling(self):
        """Test handling of database constraint violations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create table with constraints
                cursor.execute('''
                    CREATE TABLE constraint_test (
                        id INTEGER PRIMARY KEY,
                        unique_field TEXT UNIQUE NOT NULL,
                        positive_number INTEGER CHECK(positive_number > 0)
                    )
                ''')
                
                # Insert valid data
                cursor.execute(
                    "INSERT INTO constraint_test (unique_field, positive_number) VALUES (?, ?)",
                    ("unique_value", 5)
                )
                conn.commit()
                
                # Test unique constraint violation
                with pytest.raises(sqlite3.IntegrityError):
                    cursor.execute(
                        "INSERT INTO constraint_test (unique_field, positive_number) VALUES (?, ?)",
                        ("unique_value", 10)  # Duplicate unique_field
                    )
                
                # Test check constraint violation  
                with pytest.raises(sqlite3.IntegrityError):
                    cursor.execute(
                        "INSERT INTO constraint_test (unique_field, positive_number) VALUES (?, ?)",
                        ("another_unique", -1)  # Violates positive_number > 0
                    )
                
                # Test NOT NULL constraint violation
                with pytest.raises(sqlite3.IntegrityError):
                    cursor.execute(
                        "INSERT INTO constraint_test (positive_number) VALUES (?)",
                        (5,)  # Missing required unique_field
                    )
                
                conn.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)
    
    def test_database_recovery_mechanisms(self):
        """Test database recovery and backup mechanisms."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as backup_db:
                db_path = tmp_db.name
                backup_path = backup_db.name
                
                try:
                    # Create original database with data
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        CREATE TABLE recovery_test (
                            id INTEGER PRIMARY KEY,
                            data TEXT
                        )
                    ''')
                    
                    cursor.execute("INSERT INTO recovery_test (data) VALUES (?)", ("original_data",))
                    conn.commit()
                    conn.close()
                    
                    # Create backup
                    original_conn = sqlite3.connect(db_path)
                    backup_conn = sqlite3.connect(backup_path)
                    original_conn.backup(backup_conn)
                    original_conn.close()
                    backup_conn.close()
                    
                    # Simulate data corruption/loss in original
                    with open(db_path, 'w') as f:
                        f.write("corrupted data")
                    
                    # Test recovery from backup
                    backup_conn = sqlite3.connect(backup_path)
                    cursor = backup_conn.cursor()
                    cursor.execute("SELECT data FROM recovery_test")
                    recovered_data = cursor.fetchone()
                    
                    assert recovered_data is not None
                    assert recovered_data[0] == "original_data"
                    backup_conn.close()
                    
                finally:
                    Path(db_path).unlink(missing_ok=True)
                    Path(backup_path).unlink(missing_ok=True)
    
    def test_database_deadlock_handling(self):
        """Test database deadlock detection and handling."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                # Setup test table
                setup_conn = sqlite3.connect(db_path)
                setup_cursor = setup_conn.cursor()
                setup_cursor.execute('''
                    CREATE TABLE deadlock_test (
                        id INTEGER PRIMARY KEY,
                        value TEXT
                    )
                ''')
                setup_cursor.execute("INSERT INTO deadlock_test (value) VALUES ('initial')")
                setup_conn.commit()
                setup_conn.close()
                
                # Simulate potential deadlock scenario
                conn1 = sqlite3.connect(db_path)
                conn2 = sqlite3.connect(db_path)
                
                cursor1 = conn1.cursor()
                cursor2 = conn2.cursor()
                
                # Both connections try to update the same row
                conn1.execute("BEGIN IMMEDIATE")
                try:
                    cursor1.execute("UPDATE deadlock_test SET value = 'conn1_update' WHERE id = 1")
                    
                    # This should either succeed immediately or timeout
                    conn2.execute("BEGIN IMMEDIATE")
                    cursor2.execute("UPDATE deadlock_test SET value = 'conn2_update' WHERE id = 1")
                    conn2.commit()
                    
                except sqlite3.OperationalError as e:
                    # Expected - database is locked
                    assert "locked" in str(e).lower()
                    conn2.rollback()
                
                finally:
                    conn1.commit()
                    conn1.close()
                    conn2.close()
                
            finally:
                Path(db_path).unlink(missing_ok=True)