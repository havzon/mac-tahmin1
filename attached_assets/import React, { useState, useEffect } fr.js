import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const MatchPredictor = () => {
  const [data, setData] = useState([]);
  const [teams, setTeams] = useState([]);
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [odds, setOdds] = useState({
    match: { home: '', draw: '', away: '' },
    goals: { over25: '', under25: '', btts_yes: '' },
    firstHalf: { over05: '', under05: '', over15: '' }
  });

  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await window.fs.readFile('oranlar1234.csv', { encoding: 'utf8' });
        const result = Papa.parse(response, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true
        });
        
        setData(result.data);
        const uniqueTeams = [...new Set([
          ...result.data.map(match => match.HomeTeam),
          ...result.data.map(match => match.AwayTeam)
        ])].sort();
        setTeams(uniqueTeams);
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };
    loadData();
  }, []);

  const calculateFormScore = (matches, team) => {
    return matches.reduce((score, match, index) => {
      const weight = Math.exp(-index * 0.1);
      let points = 0;

      if (match.HomeTeam === team) {
        points = match.FTR === 'H' ? 3 : match.FTR === 'D' ? 1 : -1;
      } else if (match.AwayTeam === team) {
        points = match.FTR === 'A' ? 3 : match.FTR === 'D' ? 1 : -1;
      }

      return score + (points * weight);
    }, 0);
  };

  const analyzeTeamPerformance = (team) => {
    const allMatches = data.filter(m => m.HomeTeam === team || m.AwayTeam === team);
    const recentMatches = allMatches.slice(-20);
    const homeMatches = recentMatches.filter(m => m.HomeTeam === team);
    const awayMatches = recentMatches.filter(m => m.AwayTeam === team);

    const formScore = calculateFormScore(recentMatches, team);

    const seasonStats = {
      totalMatches: recentMatches.length,
      homeWins: homeMatches.filter(m => m.FTR === 'H').length,
      awayWins: awayMatches.filter(m => m.FTR === 'A').length,
      homeDraws: homeMatches.filter(m => m.FTR === 'D').length,
      awayDraws: awayMatches.filter(m => m.FTR === 'D').length,
      homeGoals: homeMatches.reduce((sum, m) => sum + m.FTHG, 0),
      awayGoals: awayMatches.reduce((sum, m) => sum + m.FTAG, 0),
      homeConceded: homeMatches.reduce((sum, m) => sum + m.FTAG, 0),
      awayConceded: awayMatches.reduce((sum, m) => sum + m.FTHG, 0)
    };

    const goalStats = {
      avgScoredHome: homeMatches.length ? seasonStats.homeGoals / homeMatches.length : 0,
      avgScoredAway: awayMatches.length ? seasonStats.awayGoals / awayMatches.length : 0,
      avgConcededHome: homeMatches.length ? seasonStats.homeConceded / homeMatches.length : 0,
      avgConcededAway: awayMatches.length ? seasonStats.awayConceded / awayMatches.length : 0,
      cleanSheets: recentMatches.filter(m => 
        (m.HomeTeam === team && m.FTAG === 0) || 
        (m.AwayTeam === team && m.FTHG === 0)
      ).length,
      failedToScore: recentMatches.filter(m => 
        (m.HomeTeam === team && m.FTHG === 0) || 
        (m.AwayTeam === team && m.FTAG === 0)
      ).length
    };

    return {
      formScore,
      seasonStats,
      goalStats,
      overallStrength: (
        (formScore / 40) * 0.5 + 
        ((seasonStats.homeWins + seasonStats.awayWins) / recentMatches.length) * 0.3 +
        (1 - (goalStats.failedToScore / recentMatches.length)) * 0.2
      )
    };
  };

  const calculatePrediction = (home, away) => {
    if (!home || !away) return null;

    const homeStats = analyzeTeamPerformance(home);
    const awayStats = analyzeTeamPerformance(away);
    const h2h = data.filter(match => 
      (match.HomeTeam === home && match.AwayTeam === away) ||
      (match.HomeTeam === away && match.AwayTeam === home)
    ).slice(-5);

    let homeWinProb, drawProb, awayWinProb;

    if (odds.match.home && odds.match.draw && odds.match.away) {
      const totalOdds = (1/parseFloat(odds.match.home) + 1/parseFloat(odds.match.draw) + 1/parseFloat(odds.match.away));
      homeWinProb = (1/parseFloat(odds.match.home)) / totalOdds;
      drawProb = (1/parseFloat(odds.match.draw)) / totalOdds;
      awayWinProb = (1/parseFloat(odds.match.away)) / totalOdds;
    } else {
      const homeStrength = homeStats.overallStrength * 1.1;
      const awayStrength = awayStats.overallStrength;
      const totalStrength = homeStrength + awayStrength;

      const formDiff = Math.abs(homeStats.formScore - awayStats.formScore);
      drawProb = 0.3 * Math.exp(-formDiff / 10);

      homeWinProb = (1 - drawProb) * (homeStrength / totalStrength);
      awayWinProb = (1 - drawProb) * (awayStrength / totalStrength);
    }

    const expectedGoals = 
      (homeStats.goalStats.avgScoredHome + awayStats.goalStats.avgConcededAway) * 0.5 +
      (awayStats.goalStats.avgScoredAway + homeStats.goalStats.avgConcededHome) * 0.5;

    let over25Prob = odds.goals.over25 ?
      1/parseFloat(odds.goals.over25) / (1/parseFloat(odds.goals.over25) + 1/parseFloat(odds.goals.under25)) :
      1 / (1 + Math.exp(2.5 - expectedGoals));

    const bttsProb = odds.goals.btts_yes ?
      1/parseFloat(odds.goals.btts_yes) / (1/parseFloat(odds.goals.btts_yes) + 1/parseFloat(odds.goals.btts_no)) :
      (1 - Math.exp(-homeStats.goalStats.avgScoredHome * awayStats.goalStats.avgScoredAway));

    return {
      homeWin: (homeWinProb * 100).toFixed(1),
      draw: (drawProb * 100).toFixed(1),
      awayWin: (awayWinProb * 100).toFixed(1),
      expectedGoals: expectedGoals.toFixed(1),
      over25: (over25Prob * 100).toFixed(1),
      btts: (bttsProb * 100).toFixed(1),
      firstHalf: {
        expectedGoals: (expectedGoals * 0.4).toFixed(1),
        over05: (1 - Math.exp(-expectedGoals * 0.4) * 100).toFixed(1)
      },
      chartData: [
        { name: 'Home Win', probability: parseFloat(homeWinProb * 100) },
        { name: 'Draw', probability: parseFloat(drawProb * 100) },
        { name: 'Away Win', probability: parseFloat(awayWinProb * 100) }
      ]
    };
  };

  const handleTeamSelect = (team, isHome) => {
    if (isHome) {
      setHomeTeam(team);
    } else {
      setAwayTeam(team);
    }
    
    if (isHome && awayTeam || !isHome && homeTeam) {
      const pred = calculatePrediction(
        isHome ? team : homeTeam,
        isHome ? awayTeam : team
      );
      setPrediction(pred);
    }
  };

  const handleOddsChange = (category, type, value) => {
    setOdds(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [type]: value
      }
    }));
    
    if (homeTeam && awayTeam) {
      const pred = calculatePrediction(homeTeam, awayTeam);
      setPrediction(pred);
    }
  };

  return (
    <div className="w-full max-w-4xl p-4">
      <h2 className="text-2xl font-bold mb-6">Advanced Match Predictor</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium mb-2">Home Team</label>
          <select 
            className="w-full p-2 border rounded"
            value={homeTeam}
            onChange={(e) => handleTeamSelect(e.target.value, true)}
          >
            <option value="">Select Home Team</option>
            {teams.map(team => (
              <option key={`home-${team}`} value={team}>{team}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Away Team</label>
          <select
            className="w-full p-2 border rounded"
            value={awayTeam}
            onChange={(e) => handleTeamSelect(e.target.value, false)}
          >
            <option value="">Select Away Team</option>
            {teams.map(team => (
              <option key={`away-${team}`} value={team}>{team}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 mb-6">
        <div className="border rounded-lg p-4">
          <h3 className="font-medium mb-3">Match Odds</h3>
          <div className="grid grid-cols-3 gap-4">
            {['home', 'draw', 'away'].map(type => (
              <div key={type}>
                <label className="block text-sm mb-1">{type.charAt(0).toUpperCase() + type.slice(1)}</label>
                <input
                  type="number"
                  step="0.01"
                  className="w-full p-2 border rounded"
                  value={odds.match[type]}
                  onChange={(e) => handleOddsChange('match', type, e.target.value)}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="border rounded-lg p-4">
          <h3 className="font-medium mb-3">Goals</h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm mb-1">Over 2.5</label>
              <input
                type="number"
                step="0.01"
                className="w-full p-2 border rounded"
                value={odds.goals.over25}
                onChange={(e) => handleOddsChange('goals', 'over25', e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Under 2.5</label>
              <input
                type="number"
                step="0.01"
                className="w-full p-2 border rounded"
                value={odds.goals.under25}
                onChange={(e) => handleOddsChange('goals', 'under25', e.target.value)}
              />
            </div>
            <div>
              <label className="block text-sm mb-1">BTTS Yes</label>
              <input
                type="number"
                step="0.01"
                className="w-full p-2 border rounded"
                value={odds.goals.btts_yes}
                onChange={(e) => handleOddsChange('goals', 'btts_yes', e.target.value)}
              />
            </div>
          </div>
        </div>
      </div>

      {prediction && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-4">Match Prediction</h3>
          <div className="h-64 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={prediction.chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="probability" fill="#3B82F6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded">
              <h4 className="font-semibold mb-2">Match Result</h4>
              <div>Home Win: {prediction.homeWin}%</div>
              <div>Draw: {prediction.draw}%</div>
              <div>Away Win: {prediction.awayWin}%</div>
            </div>
            <div className="p-4 bg-blue-50 rounded">
              <h4 className="font-semibold mb-2">Goals Prediction</h4>
              <div>Expected Goals: {prediction.expectedGoals}</div>
              <div>Over 2.5: {prediction.over25}%</div>
              <div>BTTS: {prediction.btts}%</div>
              <div>First Half Goals: {prediction.firstHalf.expectedGoals}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MatchPredictor;