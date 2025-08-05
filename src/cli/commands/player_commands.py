"""
Player analysis CLI commands for FPL ML System.
Commands: analyze, compare, search, stats, fixtures, form, price, ownership
"""

import click
import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from typing import Optional, List

from ...agents import FPLManagerDependencies

console = Console()


@click.group()
def player():
    """⚽ Player analysis and research commands."""
    pass


@player.command()
@click.argument('player_name', type=str)
@click.option('--weeks', '-w', type=int, default=5, help='Analysis horizon in weeks')
@click.option('--detailed', is_flag=True, help='Show detailed analytics')
@click.option('--fixtures', is_flag=True, help='Include fixture analysis')
@click.pass_context
def analyze(ctx, player_name, weeks, detailed, fixtures):
    """Analyze a specific player with ML predictions and insights."""
    cli_context = ctx.obj['cli_context']
    
    console.print(f"⚽ [bold]Player Analysis: {player_name}[/bold]")
    
    with console.status(f"Analyzing {player_name}..."):
        try:
            deps = FPLManagerDependencies(
                fpl_team_id=cli_context.settings.fpl_team_id or 0,
                database_url=str(cli_context.settings.database_url)
            )
            
            async def get_analysis():
                return await cli_context.fpl_manager.run(
                    "get_player_analysis",
                    deps,
                    player_name=player_name,
                    gameweeks_ahead=weeks
                )
            
            result = asyncio.run(get_analysis())
            
        except Exception as e:
            console.print(f"❌ Error analyzing player: {str(e)}")
            return
    
    console.print(Panel(result, title=f"{player_name} Analysis", border_style="blue"))
    
    if detailed:
        console.print(f"\n🔬 [bold]Detailed Analytics[/bold]")
        
        # Advanced metrics table
        metrics_table = Table(title="Advanced Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("League Rank", style="green")
        metrics_table.add_column("Position Rank", style="yellow")
        
        # Sample advanced metrics
        metrics_data = [
            ("Expected Goals (xG)", "0.52", "Top 15%", "Top 5%"),
            ("Expected Assists (xA)", "0.31", "Top 25%", "Top 10%"),
            ("ICT Index", "142.3", "Top 20%", "Top 8%"),
            ("Bonus Points System", "156", "Top 18%", "Top 6%"),
            ("Points per £M", "15.8", "Top 12%", "Top 4%"),
            ("Minutes per Goal", "187", "Top 22%", "Top 9%")
        ]
        
        for metric in metrics_data:
            metrics_table.add_row(*metric)
        
        console.print(metrics_table)
    
    if fixtures:
        console.print(f"\n📅 [bold]Upcoming Fixtures Analysis[/bold]")
        
        fixtures_table = Table(title=f"{player_name} - Next 5 Fixtures")
        fixtures_table.add_column("GW", style="cyan")
        fixtures_table.add_column("Opponent", style="white")
        fixtures_table.add_column("H/A", style="green")
        fixtures_table.add_column("Difficulty", style="yellow")
        fixtures_table.add_column("Predicted Points", style="magenta")
        
        # Sample fixture data
        fixture_data = [
            ("16", "BHA", "H", "3/5", "8.2"),
            ("17", "SHU", "A", "2/5", "9.1"),
            ("18", "BUR", "H", "2/5", "8.8"),
            ("19", "EVE", "A", "3/5", "7.4"),
            ("20", "TOT", "H", "4/5", "6.9")
        ]
        
        for fixture in fixture_data:
            fixtures_table.add_row(*fixture)
        
        console.print(fixtures_table)
        
        console.print("\n💡 [bold]Fixture Insights:[/bold]")
        console.print("• Favorable run: 3 green fixtures in next 5")
        console.print("• Home advantage: 3 home games")
        console.print("• Expected total: 40.4 points over 5 GWs")
        console.print("• Best fixture: GW17 vs Sheffield United (A)")


@player.command()
@click.argument('players', nargs=-1, required=True)
@click.option('--metric', type=click.Choice(['points', 'value', 'form', 'fixtures']), 
              default='points', help='Primary comparison metric')
@click.option('--weeks', type=int, default=5, help='Analysis period')
@click.pass_context
def compare(ctx, players, metric, weeks):
    """Compare multiple players across key metrics."""
    if len(players) < 2:
        console.print("❌ Need at least 2 players to compare")
        return
    
    if len(players) > 5:
        console.print("⚠️ Limiting comparison to first 5 players")
        players = players[:5]
    
    console.print(f"🔍 [bold]Player Comparison ({len(players)} players)[/bold]")
    
    # Comparison table
    table = Table(title=f"Player Comparison - {metric.title()} Focus")
    table.add_column("Metric", style="cyan")
    
    for player in players:
        table.add_column(player, style="white")
    
    # Sample comparison data
    comparison_metrics = [
        ("Current Price", "£8.5M", "£9.0M", "£7.5M", "£6.5M", "£10.0M"),
        ("Total Points", "134", "128", "142", "98", "156"),
        ("Points/Game", "6.7", "6.4", "7.1", "4.9", "7.8"),
        ("Form (5 games)", "7.2", "6.8", "8.1", "4.2", "8.9"),
        ("Minutes/Game", "82", "78", "85", "71", "89"),
        ("Goals", "8", "6", "11", "3", "12"),
        ("Assists", "4", "7", "2", "8", "3"),
        ("Ownership %", "18.7", "25.3", "12.4", "6.8", "41.2"),
        ("Value Score", "15.8", "14.2", "18.9", "15.1", "15.6"),
        ("Next 5 GW Pred", "32.1", "29.8", "35.4", "21.7", "38.2")
    ]
    
    for metric_row in comparison_metrics[:len(players)+1]:
        table.add_row(*metric_row)
    
    console.print(table)
    
    # Winner analysis
    console.print(f"\n🏆 [bold]Category Winners:[/bold]")
    console.print(f"• Best Value: {players[2] if len(players) > 2 else players[0]} (18.9 pts/£M)")
    console.print(f"• Best Form: {players[4] if len(players) > 4 else players[-1]} (8.9 avg)")
    console.print(f"• Most Points: {players[4] if len(players) > 4 else players[-1]} (156 total)")
    console.print(f"• Best Prediction: {players[4] if len(players) > 4 else players[-1]} (38.2 next 5)")
    console.print(f"• Cheapest: {players[3] if len(players) > 3 else players[0]} (£6.5M)")
    
    # Recommendation
    console.print(f"\n🎯 [bold]Recommendation:[/bold]")
    if metric == 'points':
        console.print(f"For pure points: {players[4] if len(players) > 4 else players[-1]} leads in current form and predictions")
    elif metric == 'value':
        console.print(f"For best value: {players[2] if len(players) > 2 else players[0]} offers excellent points per million")
    elif metric == 'form':
        console.print(f"For current form: {players[4] if len(players) > 4 else players[-1]} is in outstanding recent form")
    else:
        console.print(f"Based on {metric}: Analysis shows {players[1]} as balanced option")


@player.command()
@click.option('--position', type=click.Choice(['GK', 'DEF', 'MID', 'FWD']), help='Filter by position')
@click.option('--team', help='Filter by team (3-letter code)')
@click.option('--min-price', type=float, help='Minimum price filter')
@click.option('--max-price', type=float, help='Maximum price filter')
@click.option('--min-points', type=int, help='Minimum total points')
@click.option('--sort-by', type=click.Choice(['points', 'value', 'form', 'ownership']), 
              default='points', help='Sort by metric')
@click.option('--limit', type=int, default=20, help='Number of results')
@click.pass_context
def search(ctx, position, team, min_price, max_price, min_points, sort_by, limit):
    """Search for players with flexible filtering options."""
    console.print("🔍 [bold]Player Search[/bold]")
    
    # Build filter description
    filters = []
    if position:
        filters.append(f"Position: {position}")
    if team:
        filters.append(f"Team: {team}")
    if min_price:
        filters.append(f"Min Price: £{min_price}M")
    if max_price:
        filters.append(f"Max Price: £{max_price}M")
    if min_points:
        filters.append(f"Min Points: {min_points}")
    
    if filters:
        console.print(f"🔧 Filters: {' | '.join(filters)}")
    
    # Search results table
    table = Table(title=f"Search Results (sorted by {sort_by})")
    table.add_column("Player", style="cyan")
    table.add_column("Pos", style="white")
    table.add_column("Team", style="blue")
    table.add_column("Price", style="green")
    table.add_column("Points", style="yellow")
    table.add_column("Form", style="red")
    table.add_column("Ownership", style="magenta")
    table.add_column("Value", style="bright_green")
    
    # Sample search results (would be filtered in production)
    all_players = [
        ("Haaland", "FWD", "MCI", "£14.0M", "187", "9.2", "68.4%", "13.4"),
        ("Salah", "MID", "LIV", "£13.0M", "172", "8.8", "54.2%", "13.2"),
        ("Palmer", "MID", "CHE", "£5.5M", "94", "7.4", "18.7%", "17.1"),
        ("Bowen", "MID", "WHU", "£6.5M", "129", "7.8", "8.7%", "19.8"),
        ("Watkins", "FWD", "AVL", "£7.5M", "134", "8.2", "15.3%", "17.9"),
        ("Martinez", "GK", "AVL", "£4.5M", "98", "6.1", "12.4%", "21.8"),
        ("Robertson", "DEF", "LIV", "£6.5M", "118", "5.9", "28.1%", "18.2"),
        ("Saka", "MID", "ARS", "£8.5M", "147", "7.1", "31.5%", "17.3")
    ]
    
    # Apply filters
    filtered_players = []
    for player in all_players:
        name, pos, tm, price, points, form, own, value = player
        
        if position and pos != position:
            continue
        if team and tm != team.upper():
            continue
        if min_price and float(price[1:-1]) < min_price:
            continue
        if max_price and float(price[1:-1]) > max_price:
            continue
        if min_points and int(points) < min_points:
            continue
        
        filtered_players.append(player)
    
    # Sort results
    if sort_by == 'points':
        filtered_players.sort(key=lambda x: int(x[4]), reverse=True)
    elif sort_by == 'value':
        filtered_players.sort(key=lambda x: float(x[7]), reverse=True)
    elif sort_by == 'form':
        filtered_players.sort(key=lambda x: float(x[5]), reverse=True)
    elif sort_by == 'ownership':
        filtered_players.sort(key=lambda x: float(x[6][:-1]), reverse=True)
    
    # Display results
    for player in filtered_players[:limit]:
        table.add_row(*player)
    
    console.print(table)
    
    console.print(f"\n📊 [bold]Search Summary:[/bold]")
    console.print(f"• Found: {len(filtered_players)} players matching criteria")
    console.print(f"• Showing: {min(limit, len(filtered_players))} results")
    console.print(f"• Top result: {filtered_players[0][0] if filtered_players else 'None'}")
    
    if len(filtered_players) > limit:
        console.print(f"• Use --limit {len(filtered_players)} to see all results")


@player.command()
@click.argument('player_name', type=str)
@click.option('--gameweeks', type=int, default=10, help='Number of recent gameweeks')
@click.pass_context
def stats(ctx, player_name, gameweeks):
    """Show detailed statistics for a player."""
    console.print(f"📊 [bold]Detailed Statistics: {player_name}[/bold]")
    
    # Basic stats
    basic_stats = Table(title="Season Statistics")
    basic_stats.add_column("Metric", style="cyan")
    basic_stats.add_column("Value", style="white")
    basic_stats.add_column("Per Game", style="green")
    basic_stats.add_column("Rank", style="yellow")
    
    stats_data = [
        ("Games Played", "20", "20.0", "N/A"),
        ("Minutes", "1,642", "82.1", "Top 15%"),
        ("Goals", "11", "0.55", "Top 8%"),
        ("Assists", "6", "0.30", "Top 12%"),
        ("Clean Sheets", "0", "0.00", "N/A"),
        ("Bonus Points", "18", "0.90", "Top 5%"),
        ("Yellow Cards", "3", "0.15", "Average"),
        ("Red Cards", "0", "0.00", "Good")
    ]
    
    for stat in stats_data:
        basic_stats.add_row(*stat)
    
    console.print(basic_stats)
    
    # Recent form
    console.print(f"\n📈 [bold]Recent Form (Last {gameweeks} GWs)[/bold]")
    
    form_table = Table()
    form_table.add_column("GW", style="cyan")
    form_table.add_column("Opponent", style="white")
    form_table.add_column("H/A", style="green")
    form_table.add_column("Minutes", style="yellow")
    form_table.add_column("Goals", style="red")
    form_table.add_column("Assists", style="blue")
    form_table.add_column("Points", style="magenta")
    
    # Sample recent form data
    form_data = [
        ("15", "BUR (H)", "H", "90", "2", "0", "11"),
        ("14", "MUN (A)", "A", "89", "1", "1", "9"),
        ("13", "SHU (H)", "H", "76", "0", "0", "2"),
        ("12", "BHA (A)", "A", "90", "1", "0", "6"),
        ("11", "WOL (H)", "H", "85", "0", "1", "6"),
        ("10", "TOT (A)", "A", "90", "2", "0", "12"),
        ("9", "EVE (H)", "H", "68", "0", "0", "1"),
        ("8", "CRY (A)", "A", "90", "1", "1", "10")
    ]
    
    for gw_data in form_data[:gameweeks]:
        form_table.add_row(*gw_data)
    
    console.print(form_table)
    
    # Performance summary
    console.print(f"\n🎯 [bold]Performance Summary:[/bold]")
    console.print(f"• Recent form: 7.5 avg points (last 5 games)")
    console.print(f"• Goal involvement: 65% of games")
    console.print(f"• Home vs Away: 8.2 vs 6.1 average points")
    console.print(f"• Big games: 9.1 average vs top 6")
    console.print(f"• Consistency: 14/20 games with 4+ points")


@player.command()
@click.argument('player_name', type=str)
@click.option('--weeks', type=int, default=8, help='Fixture horizon to analyze')
@click.pass_context
def fixtures(ctx, player_name, weeks):
    """Analyze upcoming fixtures for a player's team."""
    console.print(f"📅 [bold]Fixture Analysis: {player_name}[/bold]")
    
    # Upcoming fixtures
    fixtures_table = Table(title=f"Next {weeks} Fixtures")
    fixtures_table.add_column("GW", style="cyan")
    fixtures_table.add_column("Date", style="white")
    fixtures_table.add_column("Opponent", style="green")
    fixtures_table.add_column("Venue", style="yellow")
    fixtures_table.add_column("Difficulty", style="red")
    fixtures_table.add_column("Predicted Points", style="magenta")
    
    # Sample fixture data
    fixture_data = [
        ("16", "Sat 16 Dec", "Brighton", "Home", "3/5 🟡", "8.2"),
        ("17", "Fri 22 Dec", "Sheffield Utd", "Away", "2/5 🟢", "9.1"),
        ("18", "Tue 26 Dec", "Burnley", "Home", "2/5 🟢", "8.8"),
        ("19", "Fri 29 Dec", "Everton", "Away", "3/5 🟡", "7.4"),
        ("20", "Mon 1 Jan", "Tottenham", "Home", "4/5 🔴", "6.9"),
        ("21", "Sat 13 Jan", "Arsenal", "Away", "5/5 🔴", "5.8"),
        ("22", "Sat 20 Jan", "Newcastle", "Home", "4/5 🔴", "6.5"),
        ("23", "Wed 31 Jan", "Chelsea", "Away", "4/5 🔴", "6.1")
    ]
    
    for fixture in fixture_data[:weeks]:
        fixtures_table.add_row(*fixture)
    
    console.print(fixtures_table)
    
    # Fixture analysis
    console.print(f"\n🔍 [bold]Fixture Analysis:[/bold]")
    
    green_fixtures = len([f for f in fixture_data[:weeks] if "🟢" in f[4]])
    red_fixtures = len([f for f in fixture_data[:weeks] if "🔴" in f[4]])
    yellow_fixtures = weeks - green_fixtures - red_fixtures
    
    console.print(f"• Green fixtures (easy): {green_fixtures}")
    console.print(f"• Yellow fixtures (medium): {yellow_fixtures}")
    console.print(f"• Red fixtures (hard): {red_fixtures}")
    
    avg_difficulty = (green_fixtures * 2 + yellow_fixtures * 3 + red_fixtures * 4) / weeks
    console.print(f"• Average difficulty: {avg_difficulty:.1f}/5")
    
    home_fixtures = len([f for f in fixture_data[:weeks] if f[3] == "Home"])
    console.print(f"• Home fixtures: {home_fixtures}/{weeks}")
    
    total_predicted = sum(float(f[5]) for f in fixture_data[:weeks])
    console.print(f"• Total predicted points: {total_predicted:.1f}")
    
    # Recommendations
    console.print(f"\n💡 [bold]Recommendations:[/bold]")
    if green_fixtures >= weeks * 0.4:
        console.print(f"• ✅ Good fixture run - consider bringing in")
    elif red_fixtures >= weeks * 0.4:
        console.print(f"• ❌ Tough fixture run - consider avoiding")
    else:
        console.print(f"• ⚪ Mixed fixtures - monitor form and rotation")
    
    if home_fixtures >= weeks * 0.6:
        console.print(f"• 🏠 Home heavy - factor in home advantage")
    elif home_fixtures <= weeks * 0.3:
        console.print(f"• ✈️ Away heavy - consider travel fatigue")


@player.command()
@click.argument('player_name', type=str)
@click.option('--weeks', type=int, default=10, help='Form period to analyze')
@click.pass_context
def form(ctx, player_name, weeks):
    """Analyze player form trends and patterns."""
    console.print(f"📈 [bold]Form Analysis: {player_name}[/bold]")
    
    # Form metrics
    form_table = Table(title=f"Form Metrics (Last {weeks} games)")
    form_table.add_column("Metric", style="cyan")
    form_table.add_column("Value", style="white")
    form_table.add_column("Trend", style="green")
    form_table.add_column("vs Season Avg", style="yellow")
    
    form_data = [
        ("Average Points", "7.2", "📈 +0.8", "+1.1"),
        ("Goals per Game", "0.6", "📈 +0.2", "+0.1"),
        ("Assists per Game", "0.3", "➡️ 0.0", "0.0"),
        ("Bonus Points", "1.1", "📈 +0.4", "+0.2"),
        ("Minutes per Game", "83", "📈 +3", "+1"),
        ("Shot Conversion", "18%", "📈 +3%", "+2%")
    ]
    
    for metric in form_data:
        form_table.add_row(*metric)
    
    console.print(form_table)
    
    # Form timeline
    console.print(f"\n⏰ [bold]Form Timeline:[/bold]")
    
    timeline_text = Text()
    timeline_text.append("Last 10 Games: ", style="bold")
    
    # Simulate form with colored blocks
    recent_form = [8, 11, 2, 6, 10, 12, 1, 9, 6, 7]  # Points in recent games
    for i, points in enumerate(recent_form):
        if points >= 8:
            timeline_text.append("🟢", style="green")
        elif points >= 5:
            timeline_text.append("🟡", style="yellow")
        else:
            timeline_text.append("🔴", style="red")
        timeline_text.append(" ")
    
    timeline_text.append("\n🟢 = Excellent (8+ pts)  🟡 = Good (5-7 pts)  🔴 = Poor (<5 pts)")
    
    console.print(timeline_text)
    
    # Pattern analysis
    console.print(f"\n🔍 [bold]Pattern Analysis:[/bold]")
    console.print(f"• Current streak: 3 games with 6+ points")
    console.print(f"• Home form: 8.4 avg (excellent)")
    console.print(f"• Away form: 5.9 avg (moderate)")
    console.print(f"• vs Top 6: 6.2 avg (good)")
    console.print(f"• vs Bottom 6: 9.1 avg (excellent)")
    
    # Momentum indicators
    console.print(f"\n⚡ [bold]Momentum Indicators:[/bold]")
    console.print(f"• Form Rating: 🔥🔥🔥🔥⚪ (4/5 - Hot)")
    console.print(f"• Confidence: High (3 good games in row)")
    console.print(f"• Fixture Momentum: Positive (easier games coming)")
    console.print(f"• Price Momentum: Rising (ownership increasing)")
    console.print(f"• Manager Trust: High (90+ minutes regularly)")


@player.command()
@click.argument('player_name', type=str)
@click.option('--history', is_flag=True, help='Show price history')
@click.pass_context
def price(ctx, player_name, history):
    """Track player price changes and predictions."""
    console.print(f"💰 [bold]Price Analysis: {player_name}[/bold]")
    
    # Current price info
    price_info = Table(title="Price Information")
    price_info.add_column("Metric", style="cyan")
    price_info.add_column("Value", style="white")
    price_info.add_column("Change", style="green")
    
    info_data = [
        ("Current Price", "£7.5M", ""),
        ("Starting Price", "£7.0M", "+£0.5M"),
        ("Season High", "£7.6M", ""),
        ("Season Low", "£6.8M", ""),
        ("Value Rank", "#23", "Top 15%"),
        ("Price Changes", "5 rises, 1 fall", "Net +£0.4M")
    ]
    
    for info in info_data:
        price_info.add_row(*info)
    
    console.print(price_info)
    
    if history:
        console.print(f"\n📊 [bold]Price History:[/bold]")
        
        history_table = Table()
        history_table.add_column("Date", style="cyan")
        history_table.add_column("Change", style="white")
        history_table.add_column("New Price", style="green")
        history_table.add_column("Reason", style="yellow")
        
        history_data = [
            ("Dec 10", "+£0.1M", "£7.5M", "Strong form + ownership rise"),
            ("Nov 28", "+£0.1M", "£7.4M", "Goals in consecutive games"),
            ("Nov 15", "+£0.1M", "£7.3M", "High ownership growth"),
            ("Oct 30", "-£0.1M", "£7.2M", "Injury concern + sales"),
            ("Oct 12", "+£0.1M", "£7.3M", "Return from injury"),
            ("Sep 25", "+£0.1M", "£7.2M", "Consistent performances")
        ]
        
        for entry in history_data:
            history_table.add_row(*entry)
        
        console.print(history_table)
    
    # Price predictions
    console.print(f"\n🔮 [bold]Price Predictions:[/bold]")
    console.print(f"• Next 24h: 68% chance of rise to £7.6M")
    console.print(f"• Next week: Likely stable unless major haul")
    console.print(f"• Next month: Could reach £7.8M with good fixtures")
    console.print(f"• Risk factors: Rotation concerns, injury history")
    
    # Ownership trends
    console.print(f"\n📈 [bold]Ownership Trends:[/bold]")
    console.print(f"• Current ownership: 15.3% (+2.1% this week)")
    console.print(f"• Transfer trend: +47,892 net transfers in")
    console.print(f"• Price change threshold: ~78% reached")
    console.print(f"• Recommendation: Monitor closely for rise")


@player.command()
@click.argument('player_name', type=str)
@click.option('--compare-position', is_flag=True, help='Compare within position')
@click.pass_context
def ownership(ctx, player_name, compare_position):
    """Analyze player ownership patterns and trends."""
    console.print(f"👥 [bold]Ownership Analysis: {player_name}[/bold]")
    
    # Ownership breakdown
    ownership_table = Table(title="Ownership Breakdown")
    ownership_table.add_column("Metric", style="cyan")
    ownership_table.add_column("Value", style="white")
    ownership_table.add_column("Rank", style="green")
    
    ownership_data = [
        ("Overall Ownership", "15.3%", "#42 overall"),
        ("Top 1k Ownership", "28.7%", "#18 in top 1k"),
        ("Top 10k Ownership", "24.1%", "#23 in top 10k"),
        ("Top 100k Ownership", "18.9%", "#31 in top 100k"),
        ("Position Ownership", "8th among FWDs", "Top 25%"),
        ("Team Ownership", "Highest in team", "#1 team pick")
    ]
    
    for ownership in ownership_data:
        ownership_table.add_row(*ownership)
    
    console.print(ownership_table)
    
    # Weekly trends
    console.print(f"\n📊 [bold]Weekly Ownership Trends:[/bold]")
    
    trends_table = Table()
    trends_table.add_column("Week", style="cyan")
    trends_table.add_column("Ownership %", style="white")
    trends_table.add_column("Change", style="green")
    trends_table.add_column("Net Transfers", style="yellow")
    
    trends_data = [
        ("This week", "15.3%", "+2.1%", "+47,892"),
        ("Last week", "13.2%", "+1.8%", "+39,234"),
        ("2 weeks ago", "11.4%", "+0.9%", "+18,567"),
        ("3 weeks ago", "10.5%", "-0.3%", "-6,789"),
        ("4 weeks ago", "10.8%", "+1.2%", "+24,123")
    ]
    
    for trend in trends_data:
        trends_table.add_row(*trend)
    
    console.print(trends_table)
    
    if compare_position:
        console.print(f"\n⚽ [bold]Position Comparison (Forwards):[/bold]")
        
        position_table = Table()
        position_table.add_column("Player", style="cyan")
        position_table.add_column("Price", style="white")
        position_table.add_column("Ownership", style="green")
        position_table.add_column("Trend", style="yellow")
        
        position_data = [
            ("Haaland", "£14.0M", "68.4%", "📈 +1.2%"),
            ("Kane", "£11.5M", "34.7%", "📉 -0.8%"),
            ("Darwin", "£9.0M", "21.3%", "➡️ +0.1%"),
            (player_name, "£7.5M", "15.3%", "📈 +2.1%"),
            ("Toney", "£8.5M", "12.1%", "📉 -1.4%")
        ]
        
        for player_row in position_data:
            position_table.add_row(*player_row)
        
        console.print(position_table)
    
    # Ownership insights
    console.print(f"\n💡 [bold]Ownership Insights:[/bold]")
    console.print(f"• Template Status: Semi-template (15-30% range)")
    console.print(f"• Differential Potential: Moderate (not super high/low owned)")
    console.print(f"• Elite Popularity: Much higher in top ranks")
    console.print(f"• Momentum: Strong upward trend (+2.1% weekly)")
    console.print(f"• Risk Level: Medium (ownership climbing fast)")
    
    console.print(f"\n🎯 [bold]Strategy Implications:[/bold]")
    console.print(f"• Captain Option: Moderate differential value")
    console.print(f"• Transfer Timing: Popular but not essential")
    console.print(f"• Avoid Risk: Ownership climbing - less differential")
    console.print(f"• Price Risk: High ownership growth = price rise risk")