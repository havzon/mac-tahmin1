import numpy as np
import xgboost as xgb
import logging
from typing import Dict, List, Optional, Tuple
from team_analyzer import TeamAnalyzer
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        """ML tabanlı gelişmiş gol tahmin sistemi"""
        logger.info("Initializing ML Predictor")
        self.model = None
        self.team_analyzer = TeamAnalyzer()
        self.scaler = StandardScaler()
        self._initialize_model()

    def _initialize_model(self):
        """XGBoost modelini başlat"""
        try:
            self.model = xgb.XGBRegressor(
                n_estimators=200,  # Model karmaşıklığını artır
                learning_rate=0.05,  # Daha hassas öğrenme
                max_depth=7,  # Derin ağaç yapısı
                subsample=0.8,  # Overfitting'i önle
                colsample_bytree=0.8,  # Feature seçimi
                objective='reg:squarederror'
            )
            logger.info("ML model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML model: {str(e)}")
            raise

    def predict_goals(self, match_stats: Dict, events: List[Dict], 
                     historical_data: Optional[Dict] = None) -> Dict:
        """Gelişmiş gol olasılığı tahmini"""
        try:
            # Form ve taktik analizi
            home_team_id = match_stats[0]['team']['id']
            away_team_id = match_stats[1]['team']['id']

            # Özellik çıkarımı
            features = self._extract_enhanced_features(
                match_stats, events, historical_data
            )

            # Tahmin ve güvenilirlik analizi
            prediction = self.model.predict(features)[0] if self.model else 0.5
            confidence_level, confidence_factors = self._calculate_detailed_confidence(
                features, match_stats, events
            )

            # Maç durumu ve momentum analizi
            match_state = self._analyze_match_state(events, match_stats)
            momentum_factors = self._calculate_momentum_factors(
                events, match_stats
            )

            # Tahmin ayarlama
            adjusted_prediction = self._adjust_prediction(
                prediction,
                match_state,
                momentum_factors,
                confidence_level
            )

            return {
                'prediction': adjusted_prediction,
                'confidence_level': confidence_level,
                'confidence_factors': confidence_factors,
                'match_state': match_state,
                'momentum': momentum_factors
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                'prediction': 0.5,
                'confidence_level': 'düşük',
                'confidence_factors': {},
                'match_state': {},
                'momentum': {}
            }

    def _extract_enhanced_features(self, match_stats: Dict, events: List[Dict],
                                historical_data: Optional[Dict] = None) -> np.ndarray:
        """Maç istatistiklerinden gelişmiş özellik çıkarımı"""
        try:
            features = []

            # Temel istatistikler
            basic_stats = self._extract_basic_stats(match_stats)
            features.extend(basic_stats)

            # Olay bazlı özellikler
            event_features = self._extract_event_features(events)
            features.extend(event_features)

            # Momentum faktörleri
            momentum_features = self._calculate_momentum_metrics(events)
            features.extend(momentum_features)

            # Normalize ve reshape
            features = np.array(features).reshape(1, -1)
            return self.scaler.fit_transform(features)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 15))

    def _extract_basic_stats(self, match_stats: Dict) -> List[float]:
        """Temel istatistikleri çıkar"""
        try:
            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']

            stats = []

            # Şut istatistikleri
            stats.extend([
                float(home_stats[0]['value'] or 0),  # Toplam şut
                float(away_stats[0]['value'] or 0),
                float(home_stats[2]['value'] or 0),  # İsabetli şut
                float(away_stats[2]['value'] or 0)
            ])

            # Top kontrolü
            possession = float(home_stats[9]['value'].strip('%')) / 100 if home_stats[9]['value'] else 0.5
            stats.append(possession)

            # Ataklar
            stats.extend([
                float(home_stats[13]['value'] or 0),  # Tehlikeli atak
                float(away_stats[13]['value'] or 0)
            ])

            return stats

        except Exception as e:
            logger.error(f"Error extracting basic stats: {str(e)}")
            return [0.0] * 7

    def _extract_event_features(self, events: List[Dict]) -> List[float]:
        """Olay bazlı özellikler"""
        try:
            if not events:
                return [0.0] * 5

            recent_events = events[-10:]
            event_counts = {
                'Goal': 0, 'Card': 0, 'subst': 0,
                'Var': 0, 'Shot': 0
            }

            for event in recent_events:
                event_type = event['type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

            features = [
                event_counts['Goal'] / 10,
                event_counts['Shot'] / 10,
                event_counts['Card'] / 10,
                event_counts['Var'] / 10,
                event_counts['subst'] / 10
            ]

            return features

        except Exception as e:
            logger.error(f"Error extracting event features: {str(e)}")
            return [0.0] * 5

    def _calculate_momentum_metrics(self, events: List[Dict]) -> List[float]:
        """Momentum metrikleri"""
        try:
            if not events:
                return [0.0] * 3

            recent_events = events[-5:]
            momentum = 0.0
            intensity = 0.0

            for event in recent_events:
                if event['type'] == 'Goal':
                    momentum += 0.3
                elif event['type'] == 'Shot':
                    momentum += 0.1
                elif event['type'] == 'Card':
                    momentum -= 0.1

                intensity += 0.1

            return [
                momentum,
                intensity / len(recent_events),
                len(recent_events) / 5  # Event yoğunluğu
            ]

        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {str(e)}")
            return [0.0] * 3

    def _adjust_prediction(self, base_prediction: float, match_state: Dict,
                         momentum_factors: Dict, confidence_level: str) -> float:
        """Tahmin ayarlama"""
        try:
            adjusted = base_prediction

            # Momentum etkisi
            momentum_effect = momentum_factors.get('total', 0) * 0.1
            adjusted += momentum_effect

            # Maç durumu etkisi
            if match_state.get('phase') == 'son_dakikalar':
                adjusted *= 1.2
            elif match_state.get('phase') == 'başlangıç':
                adjusted *= 0.9

            # Güven seviyesi etkisi
            confidence_multipliers = {
                'yüksek': 1.1,
                'orta': 1.0,
                'düşük': 0.9
            }
            adjusted *= confidence_multipliers.get(confidence_level, 1.0)

            return max(0.0, min(1.0, adjusted))

        except Exception as e:
            logger.error(f"Error adjusting prediction: {str(e)}")
            return base_prediction

    def _analyze_match_state(self, events: List[Dict], match_stats: Dict) -> Dict:
        """Maç durumu analizi"""
        try:
            if not events:
                return {'phase': 'başlangıç', 'intensity': 0.0}

            current_time = events[-1]['time']['elapsed']

            # Maç fazı
            if current_time <= 15:
                phase = 'başlangıç'
            elif current_time <= 75:
                phase = 'orta'
            else:
                phase = 'son_dakikalar'

            # Oyun yoğunluğu
            recent_events = events[-5:]
            intensity = len([e for e in recent_events 
                           if e['type'] in ['Shot', 'Goal', 'Corner']]) / 5

            return {
                'phase': phase,
                'intensity': intensity,
                'time': current_time
            }

        except Exception as e:
            logger.error(f"Error analyzing match state: {str(e)}")
            return {'phase': 'belirsiz', 'intensity': 0.0}

    def _calculate_momentum_factors(self, events: List[Dict], match_stats: Dict) -> Dict:
        """Momentum faktörleri hesaplama"""
        try:
            if not events:
                return {'home': 0.0, 'away': 0.0, 'total': 0.0}

            recent_events = events[-10:]
            home_momentum = 0.0
            away_momentum = 0.0

            for event in recent_events:
                event_weight = {
                    'Goal': 0.3,
                    'Shot': 0.1,
                    'Corner': 0.05,
                    'Card': -0.1
                }.get(event['type'], 0.0)

                if event['team']['id'] == match_stats[0]['team']['id']:
                    home_momentum += event_weight
                else:
                    away_momentum += event_weight

            return {
                'home': home_momentum,
                'away': away_momentum,
                'total': home_momentum + away_momentum
            }

        except Exception as e:
            logger.error(f"Error calculating momentum factors: {str(e)}")
            return {'home': 0.0, 'away': 0.0, 'total': 0.0}

    def _calculate_detailed_confidence(self, features: np.ndarray,
                                    match_stats: Dict,
                                    events: List[Dict]) -> Tuple[str, Dict]:
        """Detaylı güven analizi"""
        try:
            confidence_factors = {}

            # Veri kalitesi
            valid_stats = sum(1 for stat in match_stats[0]['statistics'] 
                            if stat['value'] is not None)
            data_quality = valid_stats / len(match_stats[0]['statistics'])

            # Maç olgunluğu
            if events:
                match_maturity = min(1.0, events[-1]['time']['elapsed'] / 90)
            else:
                match_maturity = 0.0

            # Feature varyansı
            feature_variance = float(np.var(features))

            # Toplam güven skoru
            confidence_score = (
                data_quality * 0.4 +
                match_maturity * 0.3 +
                feature_variance * 0.3
            )

            # Güven seviyesi
            if confidence_score > 0.7:
                confidence_level = 'yüksek'
            elif confidence_score > 0.4:
                confidence_level = 'orta'
            else:
                confidence_level = 'düşük'

            confidence_factors.update({
                'data_quality': data_quality,
                'match_maturity': match_maturity,
                'feature_variance': feature_variance,
                'total_score': confidence_score
            })

            return confidence_level, confidence_factors

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 'düşük', {'error': str(e)}

    def _check_statistical_consistency(self, match_stats: Dict) -> float:
        """İstatistiksel tutarlılık kontrolü"""
        try:
            consistency_scores = []
            for team_stats in match_stats:
                valid_stats = 0
                total_stats = 0

                for stat in team_stats['statistics']:
                    if stat['value'] is not None:
                        valid_stats += 1
                    total_stats += 1

                consistency_scores.append(valid_stats / max(total_stats, 1))

            return sum(consistency_scores) / len(consistency_scores)

        except Exception as e:
            logger.error(f"Error checking statistical consistency: {str(e)}")
            return 0.0

    def _check_event_consistency(self, events: List[Dict]) -> float:
        """Olay tutarlılığı kontrolü"""
        try:
            if not events:
                return 0.0

            # Son 15 dakikadaki olayları kontrol et
            recent_events = [e for e in events if e['time']['elapsed'] >= events[-1]['time']['elapsed'] - 15]

            # Olay yoğunluğu ve çeşitliliği
            event_types = set(e['type'] for e in recent_events)
            event_density = len(recent_events) / 15  # Dakika başına olay

            # Tutarlılık skoru
            type_variety = len(event_types) / 5  # Maksimum 5 farklı olay tipi beklenir
            density_score = min(1.0, event_density)  # Normalize edilmiş yoğunluk

            return (type_variety + density_score) / 2

        except Exception as e:
            logger.error(f"Error checking event consistency: {str(e)}")
            return 0.0

    def _check_momentum_quality(self, events: List[Dict]) -> float:
        """Momentum kalitesi kontrolü"""
        try:
            if not events:
                return 0.0

            recent_events = events[-10:]

            # Momentum kalitesi faktörleri
            event_spacing = []  # Olaylar arası zaman
            event_significance = []  # Olay önemi

            for i, event in enumerate(recent_events[1:], 1):
                # Olaylar arası süre
                time_diff = event['time']['elapsed'] - recent_events[i-1]['time']['elapsed']
                event_spacing.append(min(1.0, time_diff / 5))  # 5 dakika üzeri normalize

                # Olay önemi
                significance = {
                    'Goal': 1.0,
                    'Shot': 0.7,
                    'Corner': 0.5,
                    'Card': 0.3,
                    'subst': 0.2
                }.get(event['type'], 0.1)
                event_significance.append(significance)

            # Kalite skorları
            spacing_quality = sum(event_spacing) / len(event_spacing) if event_spacing else 0
            significance_quality = sum(event_significance) / len(event_significance) if event_significance else 0

            return (spacing_quality + significance_quality) / 2

        except Exception as e:
            logger.error(f"Error checking momentum quality: {str(e)}")
            return 0.0

    def _check_statistical_relevance(self, match_stats: Dict) -> float:
        """İstatistik uygunluğu kontrolü"""
        try:
            relevance_scores = []

            key_stats = ['shots_on_target', 'possession', 'dangerous_attacks']

            for team_stats in match_stats:
                stat_scores = []

                # Önemli istatistikleri kontrol et
                shots = float(team_stats['statistics'][0]['value'] or 0)
                possession = float(team_stats['statistics'][9]['value'].strip('%')) if team_stats['statistics'][9]['value'] else 50
                attacks = float(team_stats['statistics'][13]['value'] or 0)

                # İstatistik skorları
                stat_scores.extend([
                    min(1.0, shots / 5),  # 5+ şut normal
                    abs(possession - 50) / 50,  # Possession dengesi
                    min(1.0, attacks / 20)  # 20+ atak normal
                ])

                relevance_scores.append(sum(stat_scores) / len(stat_scores))

            return sum(relevance_scores) / len(relevance_scores)

        except Exception as e:
            logger.error(f"Error checking statistical relevance: {str(e)}")
            return 0.0

    def _check_basic_data_quality(self, match_stats: Dict) -> float:
        """Temel veri kalitesi kontrolü"""
        try:
            quality_score = 0.0
            required_stats = ['shots', 'possession', 'attacks']

            for team_stats in match_stats:
                for stat in team_stats['statistics']:
                    if stat['value'] is not None and stat['value'] != '':
                        quality_score += 0.1

            return min(1.0, quality_score)

        except Exception as e:
            logger.error(f"Error checking basic data quality: {str(e)}")
            return 0.0

    def _select_best_prediction(self, predictions: Dict, match_state: Dict) -> str:
        """En uygun tahmini seç"""
        try:
            # Tahmin güvenilirliklerini kontrol et
            confidence_scores = {
                level: pred.get('quality_factors', {}).get('data_quality', 0)
                for level, pred in predictions.items()
            }

            # Maç durumuna göre ağırlıklandırma
            if match_state['phase'] in ['başlangıç', 'ilk_yarı_ortası']:
                # Erken fazlarda daha muhafazakar
                confidence_scores['yüksek'] *= 0.7
                confidence_scores['orta'] *= 0.9
            elif match_state['phase'] in ['son_dakikalar']:
                # Son dakikalarda daha agresif
                confidence_scores['yüksek'] *= 1.2
                confidence_scores['orta'] *= 1.1

            # En yüksek skorlu tahmini seç
            return max(confidence_scores.items(), key=lambda x: x[1])[0]

        except Exception as e:
            logger.error(f"Error selecting best prediction: {str(e)}")
            return 'düşük'  # Varsayılan olarak düşük güven

    def _adjust_prediction(self, base_prediction: float, match_state: Dict,
                         momentum_factors: Dict, confidence_level: str) -> float:
        """Tahmin ayarlama - güven seviyesine göre"""
        try:
            adjusted = base_prediction

            # Maç fazına göre ayarlama
            phase_multipliers = {
                'başlangıç': 0.9,  # Daha temkinli
                'ilk_yarı_ortası': 1.0,
                'ilk_yarı_sonu': 1.1,  # Gol olasılığı artar
                'ikinci_yarı_başı': 0.95,
                'ikinci_yarı_ortası': 1.1,
                'son_dakikalar': 1.2  # Son dakika baskısı
            }
            adjusted *= phase_multipliers.get(match_state['phase'], 1.0)

            # Momentum etkisi
            momentum_effect = (momentum_factors['home'] - momentum_factors['away']) * 0.2
            adjusted += momentum_effect

            # Oyun yoğunluğu etkisi
            adjusted *= (1 + match_state['intensity'] * 0.1)

            # Güven seviyesi etkisi
            confidence_modifiers = {
                'yüksek': 1.1,
                'orta': 1.0,
                'düşük': 0.9
            }
            adjusted *= confidence_modifiers.get(confidence_level, 1.0)

            return max(0.0, min(1.0, adjusted))

        except Exception as e:
            logger.error(f"Error adjusting prediction: {str(e)}")
            return base_prediction

    def _extract_enhanced_features(self, match_stats: Dict, events: List[Dict],
                                 home_form: Dict, away_form: Dict,
                                 home_tactics: Dict, away_tactics: Dict) -> np.ndarray:
        """Geliştirilmiş özellik çıkarımı"""
        try:
            features = []

            # Form bazlı özellikler
            form_features = [
                home_form['metrics']['overall_form'],
                away_form['metrics']['overall_form'],
                home_form['metrics']['attack_strength'],
                away_form['metrics']['attack_strength'],
                home_form['metrics']['defense_stability'],
                away_form['metrics']['defense_stability'],
                home_form['metrics']['shooting_accuracy'],
                away_form['metrics']['shooting_accuracy']
            ]
            features.extend(form_features)

            # Taktik bazlı özellikler
            tactic_features = [
                home_tactics['tactical_rating'] / 10,
                away_tactics['tactical_rating'] / 10,
                float(home_tactics['patterns']['formation_flexibility']),
                float(away_tactics['patterns']['formation_flexibility'])
            ]
            features.extend(tactic_features)

            # Maç istatistikleri
            stats_features = self._extract_match_stats_features(match_stats)
            features.extend(stats_features)

            # Momentum ve olay bazlı özellikler
            event_features = self._extract_event_features(events)
            features.extend(event_features)

            # Normalize ve reshape
            features = np.array(features).reshape(1, -1)
            return features

        except Exception as e:
            logger.error(f"Error extracting enhanced features: {str(e)}")
            return np.zeros((1, 20))  # Genişletilmiş özellik vektörü

    def _extract_match_stats_features(self, match_stats: Dict) -> List[float]:
        """Maç istatistiklerinden özellik çıkarımı"""
        try:
            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']

            # Şut etkinliği
            home_shots = float(home_stats[2]['value'] or 0)
            home_shots_on_target = float(home_stats[0]['value'] or 0)
            away_shots = float(away_stats[2]['value'] or 0)
            away_shots_on_target = float(away_stats[0]['value'] or 0)

            # Top kontrolü
            home_possession = float(home_stats[9]['value'].strip('%')) / 100 if home_stats[9]['value'] else 0.5

            # Tehlikeli ataklar
            home_attacks = float(home_stats[13]['value'] or 0)
            away_attacks = float(away_stats[13]['value'] or 0)

            return [
                home_shots_on_target / max(home_shots, 1),
                away_shots_on_target / max(away_shots, 1),
                home_possession,
                home_attacks / max(home_attacks + away_attacks, 1),
                away_attacks / max(home_attacks + away_attacks, 1)
            ]

        except Exception as e:
            logger.error(f"Error extracting match stats features: {str(e)}")
            return [0.0] * 5

    def _extract_event_features(self, events: List[Dict]) -> List[float]:
        """Olay bazlı özellik çıkarımı"""
        try:
            if not events:
                return [0.0] * 5

            recent_events = events[-10:]  # Son 10 olay

            # Olay tipleri sayacı
            event_counts = {
                'Goal': 0, 'Card': 0, 'subst': 0,
                'Var': 0, 'Shot': 0
            }

            # Momentum hesaplama
            momentum = 0.0
            intensity = 0.0

            for event in recent_events:
                event_type = event['type']
                event_counts[event_type] = event_counts.get(event_type, 0) + 1

                # Momentum etkisi
                if event_type == 'Goal':
                    momentum += 0.3
                elif event_type == 'Shot':
                    momentum += 0.1
                elif event_type == 'Card':
                    momentum -= 0.1

                # Oyun yoğunluğu
                intensity += 0.1

            return [
                momentum,
                intensity / len(recent_events),
                event_counts['Goal'] / 10,
                event_counts['Shot'] / 10,
                event_counts['Card'] / 10
            ]

        except Exception as e:
            logger.error(f"Error extracting event features: {str(e)}")
            return [0.0] * 5

    def _analyze_match_state(self, events: List[Dict], match_stats: Dict) -> Dict:
        """Detaylı maç durumu analizi"""
        try:
            if not events:
                return {'phase': 'başlangıç', 'intensity': 0.0, 'control': 'dengeli'}

            current_time = events[-1]['time']['elapsed']
            recent_events = events[-5:]

            # Maç fazı belirleme
            if current_time <= 15:
                phase = 'başlangıç'
            elif current_time <= 35:
                phase = 'ilk_yarı_ortası'
            elif current_time <= 45:
                phase = 'ilk_yarı_sonu'
            elif current_time <= 60:
                phase = 'ikinci_yarı_başı'
            elif current_time <= 80:
                phase = 'ikinci_yarı_ortası'
            else:
                phase = 'son_dakikalar'

            # Oyun yoğunluğu
            intensity = len([e for e in recent_events 
                           if e['type'] in ['Shot', 'Goal', 'Corner']]) / 5

            # Oyun kontrolü
            home_possession = float(match_stats[0]['statistics'][9]['value'].strip('%')) / 100 \
                            if match_stats[0]['statistics'][9]['value'] else 0.5

            if home_possession > 0.6:
                control = 'ev_sahibi'
            elif home_possession < 0.4:
                control = 'deplasman'
            else:
                control = 'dengeli'

            return {
                'phase': phase,
                'intensity': intensity,
                'control': control,
                'time': current_time
            }

        except Exception as e:
            logger.error(f"Error analyzing match state: {str(e)}")
            return {'phase': 'belirsiz', 'intensity': 0.0, 'control': 'dengeli'}

    def _calculate_momentum_factors(self, events: List[Dict], match_stats: Dict) -> Dict:
        """Gelişmiş momentum faktörleri hesaplama"""
        try:
            if not events:
                return {'home': 0.0, 'away': 0.0, 'trend': 'dengeli'}

            # Son olayların analizi
            recent_events = events[-10:]
            home_momentum = 0.0
            away_momentum = 0.0

            for event in recent_events:
                event_weight = {
                    'Goal': 0.3,
                    'Shot': 0.1,
                    'Corner': 0.05,
                    'Card': -0.1,
                    'Var': 0.05,
                    'subst': 0.02
                }.get(event['type'], 0.0)


                # Form bazlı ağırlık ayarı  (removed because home_form and away_form are not passed)

                if event['team']['id'] == match_stats[0]['team']['id']:
                    home_momentum += event_weight
                else:
                    away_momentum += event_weight

            # Momentum trendi
            if home_momentum > away_momentum * 1.5:
                trend = 'güçlü_ev_sahibi'
            elif away_momentum > home_momentum * 1.5:
                trend = 'güçlü_deplasman'
            elif home_momentum > away_momentum:
                trend = 'hafif_ev_sahibi'
            elif away_momentum > home_momentum:
                trend = 'hafif_deplasman'
            else:
                trend = 'dengeli'

            return {
                'home': home_momentum,
                'away': away_momentum,
                'trend': trend,
                'total': home_momentum + away_momentum
            }

        except Exception as e:
            logger.error(f"Error calculating momentum factors: {str(e)}")
            return {'home': 0.0, 'away': 0.0, 'trend': 'dengeli'}


    def _calculate_detailed_confidence(self, features: np.ndarray, match_stats: Dict,
                                    events: List[Dict], home_form: Dict,
                                    away_form: Dict) -> Tuple[str, Dict]:
        """Detaylı güven analizi"""
        try:
            confidence_factors = {}

            # Veri kalitesi
            data_quality = self._check_data_quality(match_stats)
            confidence_factors['data_quality'] = data_quality

            # Form güvenilirliği
            form_reliability = (
                home_form['metrics']['overall_form'] +
                away_form['metrics']['overall_form']
            ) / 2
            confidence_factors['form_reliability'] = form_reliability

            # Maç olgunluğu
            if events:
                match_maturity = min(1.0, events[-1]['time']['elapsed'] / 90)
            else:
                match_maturity = 0.0
            confidence_factors['match_maturity'] = match_maturity

            # Feature varyansı
            feature_variance = float(np.var(features))
            confidence_factors['feature_variance'] = feature_variance

            # Toplam güven skoru
            confidence_score = (
                data_quality * 0.3 +
                form_reliability * 0.3 +
                match_maturity * 0.2 +
                feature_variance * 0.2
            )

            # Güven seviyesi belirleme
            if confidence_score > 0.7:
                confidence_level = 'yüksek'
            elif confidence_score > 0.4:
                confidence_level = 'orta'
            else:
                confidence_level = 'düşük'

            return confidence_level, confidence_factors

        except Exception as e:
            logger.error(f"Error calculating detailed confidence: {str(e)}")
            return 'düşük', {'error': str(e)}

    def _predict_detailed_goal_time(self, events: List[Dict], goal_prob: float,
                                  momentum_factors: Dict) -> Tuple[Optional[int], Dict]:
        """Detaylı gol zamanı tahmini"""
        try:
            if not events:
                return None, {}

            current_time = events[-1]['time']['elapsed']

            # Temel zaman aralığı
            if goal_prob > 0.7:
                base_range = (5, 15)  # Yüksek olasılık
            elif goal_prob > 0.4:
                base_range = (10, 25)  # Orta olasılık
            else:
                base_range = (15, 35)  # Düşük olasılık

            # Momentum etkisi
            momentum_effect = momentum_factors['total'] * 5
            adjusted_min = max(1, int(base_range[0] - momentum_effect))
            adjusted_max = max(adjusted_min + 5, int(base_range[1] - momentum_effect))

            # Maç fazına göre ayarlama
            if current_time >= 80:
                adjusted_max = min(adjusted_max, 90 - current_time)
                adjusted_min = min(adjusted_min, adjusted_max - 2)

            # Tahmin ve faktörler
            predicted_time = current_time + np.random.randint(adjusted_min, adjusted_max)
            time_factors = {
                'base_range': base_range,
                'momentum_effect': momentum_effect,
                'adjusted_range': (adjusted_min, adjusted_max),
                'current_time': current_time
            }

            return min(90, predicted_time), time_factors

        except Exception as e:
            logger.error(f"Error predicting detailed goal time: {str(e)}")
            return None, {'error': str(e)}

    def extract_features(self, match_stats: Dict, events: List[Dict], historical_data: Optional[Dict] = None) -> np.ndarray:
        """Maç istatistiklerinden gelişmiş özellik çıkarımı"""
        try:
            home_stats = match_stats[0]['statistics']
            away_stats = match_stats[1]['statistics']
            home_team = match_stats[0]['team']['name']
            away_team = match_stats[1]['team']['name']

            features = []

            # Temel istatistikler ve özelleştirilmiş ağırlıklar
            team_weights = self._calculate_team_weights(home_team, away_team, historical_data)

            # Şut ve gol şansı metrikleri
            shot_metrics = self._calculate_shot_metrics(home_stats, away_stats)
            features.extend([
                shot_metrics['home_shot_efficiency'] * team_weights['home_attack'],
                shot_metrics['away_shot_efficiency'] * team_weights['away_attack'],
                shot_metrics['home_conversion_rate'] * team_weights['home_finish'],
                shot_metrics['away_conversion_rate'] * team_weights['away_finish']
            ])

            # Taktiksel metrikler
            tactical_metrics = self._calculate_tactical_metrics(home_stats, away_stats)
            features.extend([
                tactical_metrics['home_possession_effectiveness'] * team_weights['home_control'],
                tactical_metrics['away_possession_effectiveness'] * team_weights['away_control'],
                tactical_metrics['home_pressing_success'] * team_weights['home_press'],
                tactical_metrics['away_pressing_success'] * team_weights['away_press']
            ])

            # Momentum ve form analizi
            momentum_metrics = self._calculate_momentum_metrics(events, home_team, away_team)
            features.extend([
                momentum_metrics['home_momentum'] * team_weights['home_momentum'],
                momentum_metrics['away_momentum'] * team_weights['away_momentum'],
                momentum_metrics['home_energy'],
                momentum_metrics['away_energy']
            ])

            # Maç durumu ve zaman bazlı faktörler
            time_metrics = self._calculate_time_based_metrics(events)
            features.extend([
                time_metrics['game_intensity'],
                time_metrics['time_pressure'],
                time_metrics['goal_probability_modifier']
            ])

            # Normalize features
            features = np.array(features).reshape(1, -1)
            return self.scaler.fit_transform(features)

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return np.zeros((1, 15))  # Genişletilmiş özellik vektörü

    def _calculate_team_weights(self, home_team: str, away_team: str, historical_data: Optional[Dict]) -> Dict:
        """Takım bazlı ağırlıkları hesapla"""
        try:
            # Varsayılan ağırlıklar
            weights = {
                'home_attack': 1.0, 'away_attack': 1.0,
                'home_finish': 1.0, 'away_finish': 1.0,
                'home_control': 1.0, 'away_control': 1.0,
                'home_press': 1.0, 'away_press': 1.0,
                'home_momentum': 1.0, 'away_momentum': 1.0
            }

            if historical_data:
                # Geçmiş performansa dayalı ağırlık ayarlaması
                home_form = historical_data.get(home_team, {}).get('form', 0.5)
                away_form = historical_data.get(away_team, {}).get('form', 0.5)

                weights['home_attack'] *= (1 + home_form)
                weights['away_attack'] *= (1 + away_form)
                weights['home_finish'] *= (1 + home_form * 0.5)
                weights['away_finish'] *= (1 + away_form * 0.5)

            return weights

        except Exception as e:
            logger.error(f"Error calculating team weights: {str(e)}")
            return dict.fromkeys(weights.keys(), 1.0)

    def _calculate_shot_metrics(self, home_stats: List[Dict], away_stats: List[Dict]) -> Dict:
        """Şut ve gol şansı metriklerini hesapla"""
        try:
            # Ev sahibi metrikleri
            home_shots = float(home_stats[2]['value'] or 0)
            home_shots_on_target = float(home_stats[0]['value'] or 0)
            home_dangerous_attacks = float(home_stats[13]['value'] or 0)

            # Deplasman metrikleri
            away_shots = float(away_stats[2]['value'] or 0)
            away_shots_on_target = float(away_stats[0]['value'] or 0)
            away_dangerous_attacks = float(away_stats[13]['value'] or 0)

            return {
                'home_shot_efficiency': home_shots_on_target / max(home_shots, 1),
                'away_shot_efficiency': away_shots_on_target / max(away_shots, 1),
                'home_conversion_rate': home_dangerous_attacks / max(home_shots * 2, 1),
                'away_conversion_rate': away_dangerous_attacks / max(away_shots * 2, 1)
            }

        except Exception as e:
            logger.error(f"Error calculating shot metrics: {str(e)}")
            return dict.fromkeys(['home_shot_efficiency', 'away_shot_efficiency',
                                'home_conversion_rate', 'away_conversion_rate'], 0.0)

    def _calculate_tactical_metrics(self, home_stats: List[Dict], away_stats: List[Dict]) -> Dict:
        """Taktiksel metrikleri hesapla"""
        try:
            # Possession ve kontrol metrikleri
            home_possession = float(home_stats[9]['value'].strip('%')) / 100 if home_stats[9]['value'] else 0.5
            away_possession = 1 - home_possession

            # Pressing ve agresiflik
            home_fouls = float(home_stats[7]['value'] or 0)
            away_fouls = float(away_stats[7]['value'] or 0)

            # Taktik etkinliği
            home_passes = float(home_stats[12]['value'] or 0)
            away_passes = float(away_stats[12]['value'] or 0)

            return {
                'home_possession_effectiveness': home_possession * (home_passes / max(home_passes + away_passes, 1)),
                'away_possession_effectiveness': away_possession * (away_passes / max(home_passes + away_passes, 1)),
                'home_pressing_success': 1 - (away_passes / max(home_passes + away_passes, 1)),
                'away_pressing_success': 1 - (home_passes / max(home_passes + away_passes, 1))
            }

        except Exception as e:
            logger.error(f"Error calculating tactical metrics: {str(e)}")
            return dict.fromkeys(['home_possession_effectiveness', 'away_possession_effectiveness',
                                'home_pressing_success', 'away_pressing_success'], 0.0)

    def _calculate_momentum_metrics(self, events: List[Dict], home_team: str, away_team: str) -> Dict:
        """Momentum ve enerji metriklerini hesapla"""
        try:
            metrics = {
                'home_momentum': 0.0, 'away_momentum': 0.0,
                'home_energy': 1.0, 'away_energy': 1.0
            }

            if not events:
                return metrics

            # Son olayların analizi
            recent_events = events[-10:]  # Son 10 olay
            for event in recent_events:
                event_team = event['team']['name']
                event_time = event['time']['elapsed']

                # Olay ağırlıkları
                weights = {
                    'Goal': 0.3,
                    'Card': -0.1,
                    'subst': 0.05,
                    'Var': 0.05,
                    'Shot': 0.1
                }

                # Momentum hesaplama
                event_weight = weights.get(event['type'], 0.0)
                if event_team == home_team:
                    metrics['home_momentum'] += event_weight
                    metrics['home_energy'] *= (1 - event_time/90 * 0.3)  # Enerji düşüşü
                else:
                    metrics['away_momentum'] += event_weight
                    metrics['away_energy'] *= (1 - event_time/90 * 0.3)

            # Normalize momentum
            total_momentum = abs(metrics['home_momentum']) + abs(metrics['away_momentum'])
            if total_momentum > 0:
                metrics['home_momentum'] /= total_momentum
                metrics['away_momentum'] /= total_momentum

            return metrics

        except Exception as e:
            logger.error(f"Error calculating momentum metrics: {str(e)}")
            return dict.fromkeys(['home_momentum', 'away_momentum', 'home_energy', 'away_energy'], 0.0)

    def _calculate_time_based_metrics(self, events: List[Dict]) -> Dict:
        """Zaman bazlı metrikleri hesapla"""
        try:
            if not events:
                return {'game_intensity': 0.5, 'time_pressure': 0.0, 'goal_probability_modifier': 1.0}

            current_time = events[-1]['time']['elapsed']

            # Oyun yoğunluğu
            recent_events = len([e for e in events if e['time']['elapsed'] > max(0, current_time - 15)])
            game_intensity = min(1.0, recent_events / 10)

            # Zaman baskısı
            time_pressure = min(1.0, max(0.0, (current_time - 60) / 30)) if current_time > 60 else 0.0

            # Gol olasılığı modifiyesi
            if current_time >= 85:
                goal_prob_mod = 1.5  # Son dakika baskısı
            elif current_time >= 70:
                goal_prob_mod = 1.2  # Oyun sonu yaklaşıyor
            else:
                goal_prob_mod = 1.0

            return {
                'game_intensity': game_intensity,
                'time_pressure': time_pressure,
                'goal_probability_modifier': goal_prob_mod
            }

        except Exception as e:
            logger.error(f"Error calculating time-based metrics: {str(e)}")
            return {'game_intensity': 0.5, 'time_pressure': 0.0, 'goal_probability_modifier': 1.0}

    def _calculate_confidence(self, features: np.ndarray, match_stats: Dict) -> str:
        """Geliştirilmiş güven seviyesi hesaplama"""
        try:
            # Feature değerlerinin varyansı
            feature_variance = np.var(features)

            # Veri kalitesi kontrolü
            data_quality = self._check_data_quality(match_stats)

            # Güven skoru hesaplama
            confidence_score = (feature_variance * 0.4 + data_quality * 0.6)

            if confidence_score > 0.7:
                return 'yüksek'
            elif confidence_score > 0.4:
                return 'orta'
            else:
                return 'düşük'

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 'düşük'

    def _check_data_quality(self, match_stats: Dict) -> float:
        """Veri kalitesi kontrolü"""
        try:
            total_fields = 0
            valid_fields = 0

            for team_stats in match_stats:
                for stat in team_stats['statistics']:
                    total_fields += 1
                    if stat['value'] is not None and stat['value'] != '':
                        valid_fields += 1

            return valid_fields / max(total_fields, 1)

        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            return 0.0

    def _predict_goal_time(self, events: List[Dict], goal_prob: float) -> Optional[int]:
        """Geliştirilmiş gol zamanı tahmini"""
        try:
            if not events:
                return None

            current_time = events[-1]['time']['elapsed']

            # Oyun temposu ve yoğunluğu
            time_metrics = self._calculate_time_based_metrics(events)
            game_intensity = time_metrics['game_intensity']

            # Olasılığa göre zaman aralığı belirleme
            if goal_prob > 0.7:
                base_range = (5, 15)  # Yüksek olasılık: yakın zamanda
            elif goal_prob > 0.4:
                base_range = (10, 25)  # Orta olasılık
            else:
                base_range = (15, 35)  # Düşük olasılık: daha uzun süre

            # Tempo bazlı ayarlama
            min_time = max(1, int(base_range[0] * (1 - game_intensity * 0.3)))
            max_time = max(min_time + 5, int(base_range[1] * (1 - game_intensity * 0.3)))

            predicted_time = current_time + np.random.randint(min_time, max_time)
            return min(90, predicted_time)

        except Exception as e:
            logger.error(f"Error predicting goal time: {str(e)}")
            return None

    def scaler(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        return self.scaler