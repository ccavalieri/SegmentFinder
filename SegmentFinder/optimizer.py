import pandas as pd
import numpy as np
import itertools
from scipy import stats
from tqdm import tqdm

class BruteForceFinder:
    """
    Brute force optimizer that tests all possible combinations of given conditions 
    and finds the optimal segment from an A/B test

    """
    
    def __init__(self, min_uplift=0.05, min_segment_size=1000, significance_level=0.05):
        """
        Parameters:
        -----------
        min_uplift : float
            Minimal uplift required (ex: 0.05 = 5%)
        min_segment_size : int
            Minimal segment size
        significance_level : float
            A/B test significance level (default: 0.05)
        """
        self.min_uplift = min_uplift
        self.min_segment_size = min_segment_size
        self.significance_level = significance_level
    
    def calculate_uplift_with_significance(self, df_segment):
        """
        Calculates uplift as (conv_teste/conv_controle)-1 and test
        
        Returns:
        --------
        tuple: (uplift, test_conv, control_conv, is_significant, p_value)
        """
        if len(df_segment) == 0:
            return 0, 0, 0, False, 1.0
            
        test_group = df_segment[df_segment['group'] == 'test']
        control_group = df_segment[df_segment['group'] == 'control']
        
        if len(test_group) == 0 or len(control_group) == 0:
            return 0, 0, 0, False, 1.0
            
        test_conv = test_group['converted'].mean()
        control_conv = control_group['converted'].mean()
        
        # Avoids zero division
        if control_conv == 0:
            if test_conv > 0:
                return float('inf'), test_conv, control_conv, False, 1.0
            else:
                return 0, test_conv, control_conv, False, 1.0
        
        # Calculates uplift
        uplift = (test_conv / control_conv) - 1
        
        # Significance test
        n1, n2 = len(test_group), len(control_group)
        x1, x2 = test_group['converted'].sum(), control_group['converted'].sum()
        
        if n1 < 30 or n2 < 30:  # Amostra muito pequena
            return uplift, test_conv, control_conv, False, 1.0
        
        # Z-test
        p1, p2 = x1/n1, x2/n2
        p_pooled = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        if se == 0:
            return uplift, test_conv, control_conv, False, 1.0
            
        z = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        is_significant = p_value < self.significance_level
        
        return uplift, test_conv, control_conv, is_significant, p_value

    def calculate_overall_uplift(self, df):
        """
        Calculates overall results for the A/B test
        
        Returns:
        --------
        dict: Uplift, p-value and other test infos
        """
        test_group = df[df['group'] == 'test']
        control_group = df[df['group'] == 'control']
        
        if len(test_group) == 0 or len(control_group) == 0:
            return {
                'uplift': 0,
                'test_conversion': 0,
                'control_conversion': 0,
                'test_size': len(test_group),
                'control_size': len(control_group),
                'is_significant': False,
                'p_value': 1.0,
                'confidence_interval': (0, 0)
            }
        
        test_conv = test_group['converted'].mean()
        control_conv = control_group['converted'].mean()
        
        # Avoids zero division
        if control_conv == 0:
            if test_conv > 0:
                uplift = float('inf')
            else:
                uplift = 0
        else:
            uplift = (test_conv / control_conv) - 1
        
        # Significance test
        n1, n2 = len(test_group), len(control_group)
        x1, x2 = test_group['converted'].sum(), control_group['converted'].sum()
        
        # Z-test
        p1, p2 = x1/n1, x2/n2
        p_pooled = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        
        if se == 0:
            z = 0
            p_value = 1.0
            is_significant = False
            ci_lower, ci_upper = 0, 0
        else:
            z = (p1 - p2) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            is_significant = p_value < self.significance_level
            
            # Confidence interval
            diff = p1 - p2
            margin_error = 1.96 * se  # 95% confidence
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
        
        return {
            'uplift': uplift,
            'test_conversion': test_conv,
            'control_conversion': control_conv,
            'test_size': n1,
            'control_size': n2,
            'is_significant': is_significant,
            'p_value': p_value,
            'z_score': z if 'z' in locals() else 0,
            'confidence_interval': (ci_lower, ci_upper),
            'difference': test_conv - control_conv
        }

    def print_overall_stats(self, df):
        """
        Prints stats from the A/B test
        """
        overall_stats = self.calculate_overall_uplift(df)
        
        print(f"\n{'='*80}")
        print(f"ESTATÍSTICAS GERAIS DO TESTE A/B")
        print(f"{'='*80}")
        
        print(f"📊 Tamanhos dos grupos:")
        print(f"   Teste: {overall_stats['test_size']:,} usuários")
        print(f"   Controle: {overall_stats['control_size']:,} usuários")
        print(f"   Total: {overall_stats['test_size'] + overall_stats['control_size']:,} usuários")
        
        print(f"\n🎯 Conversões:")
        print(f"   Teste: {overall_stats['test_conversion']:.2%}")
        print(f"   Controle: {overall_stats['control_conversion']:.2%}")
        print(f"   Diferença Absoluta: {overall_stats['difference']:.2%}")
        
        if overall_stats['uplift'] == float('inf'):
            print(f"   🚀 UPLIFT GERAL: ∞ (controle = 0%)")
        else:
            print(f"   🚀 UPLIFT GERAL: {overall_stats['uplift']:.2%}")
        
        print(f"\n📈 Significância Estatística:")
        #print(f"   Z-score: {overall_stats['z_score']:.4f}")
        print(f"   P-valor: {overall_stats['p_value']:.6f}")
        
        if overall_stats['is_significant']:
            print(f"   ✅ RESULTADO SIGNIFICATIVO (p < {self.significance_level})")
        else:
            print(f"   ❌ RESULTADO NÃO SIGNIFICATIVO (p >= {self.significance_level})")
        
        ci_lower, ci_upper = overall_stats['confidence_interval']
        print(f"   Intervalo de Confiança 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Additional convertions
        if overall_stats['difference'] > 0:
            total_additional = overall_stats['test_size'] * overall_stats['difference']
            print(f"\n💰 Conversões Adicionais Totais: {total_additional:.1f}")
        
        return overall_stats

    def generate_feature_conditions(self, df, features, numeric_percentiles=None, custom_thresholds=None):
        """
        Generates all possible conditions for all features
        
        Parameters:
        -----------
        numeric_percentiles : list, optional
            Percentiles for numeric features (default: [25, 50, 75])
        custom_thresholds : dict, optional
            Custom thresholds per feature. Ex: {'age': [18, 25, 35, 50]}
        """
        if numeric_percentiles is None:
            numeric_percentiles = [25, 50, 75]
        
        if custom_thresholds is None:
            custom_thresholds = {}
            
        conditions_by_feature = {}
        
        print(f"Gerando condições para {len(features)} features:")
        
        for feature in features:
            conditions = []
            
            if df[feature].dtype in ['object', 'category', 'bool']:
                # Categorical features: two oprtion for each feature
                unique_values = df[feature].unique()
                for value in unique_values:
                    conditions.append((feature, '==', value))
                print(f"  {feature} (categórica): {len(unique_values)} valores únicos")
                
            else:
                # Numerical features: uses custom thresholds or percentiles
                if feature in custom_thresholds:
                    # Uses custom thresholds
                    thresholds = custom_thresholds[feature]
                    print(f"  {feature} (numérica): thresholds customizados {thresholds}")
                else:
                    # Uses percentiles
                    thresholds = [df[feature].quantile(p/100) for p in numeric_percentiles]
                    thresholds = list(set(thresholds))
                    print(f"  {feature} (numérica): percentis {numeric_percentiles} = {[f'{t:.2f}' for t in thresholds]}")
                
                # Creates conditions >= and < for each threshold
                for threshold in thresholds:
                    conditions.extend([
                        (feature, '>=', threshold),
                        (feature, '<', threshold)
                    ])
            
            conditions_by_feature[feature] = conditions
            #print(f"    Total de condições para {feature}: {len(conditions)}")
        
        return conditions_by_feature

    def apply_conditions(self, df, conditions):
        """
       Apply conditions to DataFrame
        """
        mask = pd.Series([True] * len(df), index=df.index)
        
        for feature, operator, value in conditions:
            if operator == '==':
                mask &= (df[feature] == value)
            elif operator == '>=':
                mask &= (df[feature] >= value)
            elif operator == '<':
                mask &= (df[feature] < value)
            elif operator == '>':
                mask &= (df[feature] > value)
            elif operator == '<=':
                mask &= (df[feature] <= value)
        
        return df[mask]

    def brute_force_search(self, df, features, max_conditions=3, max_combinations=10000, 
                          numeric_percentiles=None, custom_thresholds=None, show_progress=True,
                          objective='additional_conversions'):
        """
        Brute force search that tests every combination
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns 'group' (test/control) and 'converted' (0/1)
        features : list
            List of features to be tested
        max_conditions : int
            Max number of conditions to consider at each segment
        max_combinations : int
            Max number of combinations to test (to control time)
        numeric_percentiles : list, optional
            List of numeric percentiles
        custom_thresholds : dict, optional
            Custom thresholds per feature
        show_progress : bool, optional
            If the progress bar has to be displayed (default: True)
        objective: : string, optional
            The objective that has to be optimize: additional_convertions (dafault), 
            segment_size or uplift
        
        Returns:
        --------
        list: List of segments ordered by the objective
        """
        print(f"Iniciando busca por força bruta...")
        
        # Prints overall statistics
        overall_stats = self.print_overall_stats(df)
        
        print(f"\n{'='*80}")
        print(f"CONFIGURAÇÃO DA BUSCA")
        print(f"{'='*80}")
        print(f"- Features: {features}")
        print(f"- Uplift mínimo: {self.min_uplift:.1%}")
        print(f"- Tamanho mínimo do segmento: {self.min_segment_size}")
        print(f"- Máximo de {max_conditions} condições por segmento")
        print(f"- Limite de combinações: {max_combinations:,}")
        
        # Generates all possible conditions
        conditions_by_feature = self.generate_feature_conditions(
            df, features, numeric_percentiles, custom_thresholds
        )
        
        # Lists all possible conditions
        all_conditions = []
        for feature, conditions in conditions_by_feature.items():
            all_conditions.extend(conditions)
        
        #print(f"\nTotal de condições individuais: {len(all_conditions)}")
        
        # Calculates the number of possible conditions
        total_theoretical_combinations = 0
        for num_conditions in range(1, max_conditions + 1):
            from math import comb
            combinations_this_level = comb(len(all_conditions), num_conditions)
            total_theoretical_combinations += combinations_this_level
            #print(f"- {num_conditions} condição(ões): {combinations_this_level:,} combinações teóricas")
        
        #print(f"- Total teórico: {total_theoretical_combinations:,} combinações")
        #print(f"- Limite prático: {min(max_combinations, total_theoretical_combinations):,} combinações")
        
        best_segments = []
        combinations_tested = 0
        
        # Calculates the number of combinations actually tested
        actual_combinations_to_test = 0
        for num_conditions in range(1, max_conditions + 1):
            from math import comb
            combinations_this_level = comb(len(all_conditions), num_conditions)
            actual_combinations_to_test += min(combinations_this_level, max_combinations - actual_combinations_to_test)
            if actual_combinations_to_test >= max_combinations:
                actual_combinations_to_test = max_combinations
                break
        
        print(f"- Combinações que serão testadas: {actual_combinations_to_test:,}")
        
        # Configures progress bar
        progress_bar = None
        if show_progress:
            try:
                # Use fix total
                progress_bar = tqdm(
                    total=actual_combinations_to_test,
                    desc="Buscando segmentos",
                    unit=" comb",
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}"
                )
            except Exception as e:
                print(f"Aviso: Não foi possível inicializar barra de progresso: {e}")
                print("Continuando sem barra de progresso...")
                show_progress = False
                progress_bar = None
        
        # Statistical info
        stats = {
            'total_generated': 0,
            'conflicts_skipped': 0,
            'too_small': 0,
            'insufficient_groups': 0,
            'low_uplift': 0,
            'valid_segments': 0
        }
        
        # Tests all combinations
        for num_conditions in range(1, max_conditions + 1):
            if not show_progress:
                print(f"\nTestando combinações com {num_conditions} condição(ões)...")
            
            combinations_for_this_level = 0
            
            # Generates all combinations
            for conditions in itertools.combinations(all_conditions, num_conditions):
                if combinations_tested >= max_combinations:
                    if show_progress and progress_bar is not None:
                        progress_bar.set_description("✅ Limite atingido")
                    else:
                        print(f"  Limite de {max_combinations:,} combinações atingido.")
                    break
                
                stats['total_generated'] += 1
                
                # Checks if there are combinations with conflicts
                features_in_conditions = [c[0] for c in conditions]
                if len(features_in_conditions) != len(set(features_in_conditions)):
                    stats['conflicts_skipped'] += 1
                    continue  
                
                combinations_tested += 1
                combinations_for_this_level += 1
                
                # Updates progress bar
                if show_progress and progress_bar is not None:
                    progress_bar.update(1)                    
                    if combinations_tested % 100 == 0:
                        valid_rate = (stats['valid_segments'] / combinations_tested * 100) if combinations_tested > 0 else 0
                        postfix = f"Válidos: {stats['valid_segments']} ({valid_rate:.1f}%)"
                        if len(best_segments) > 0:
                            postfix += f", Melhor: {best_segments[0]['uplift']:.1%}"
                        progress_bar.set_postfix_str(postfix)
                
                # Apply conditions and creates segment
                segment = self.apply_conditions(df, conditions)
                
                # Progress w/o bar
                if not show_progress and combinations_tested % 1000 == 0:
                    print(f"    Testadas: {combinations_tested:,} combinações...")
                               
                if len(segment) < self.min_segment_size:
                    stats['too_small'] += 1
                    continue
                            
                test_count = len(segment[segment['group'] == 'test'])
                control_count = len(segment[segment['group'] == 'control'])
                
                if test_count < 30 or control_count < 30:
                    stats['insufficient_groups'] += 1
                    continue
                
                # Calculates uplift
                uplift, test_conv, control_conv, is_sig, p_val = self.calculate_uplift_with_significance(segment)
                
                if uplift < self.min_uplift:
                    stats['low_uplift'] += 1
                    continue
                
                stats['valid_segments'] += 1
                
                # Calculates additional convertions
                additional_conversions = test_count * (test_conv - control_conv)
                
                # Adds to best segments list
                best_segments.append({
                    'conditions': conditions,
                    'segment_size': len(segment),
                    'test_size': test_count,
                    'control_size': control_count,
                    'uplift': uplift,
                    'test_conversion': test_conv,
                    'control_conversion': control_conv,
                    'additional_conversions': additional_conversions,
                    'is_significant': is_sig,
                    'p_value': p_val,
                    'num_conditions': num_conditions
                })
                
                # Sort to keep the 50 best segments
                if len(best_segments) > 100:
                    best_segments.sort(key=lambda x: x[objective], reverse=True)
                    best_segments = best_segments[:50]  
            
            if not show_progress:
                print(f"  Combinações testadas neste nível: {combinations_for_this_level:,}")
            
            if combinations_tested >= max_combinations:
                break
        
        if show_progress and progress_bar is not None:
            progress_bar.set_description("✅ Busca concluída")
            progress_bar.close()
        
        print(f"\n{'='*80}")
        print(f"ESTATÍSTICAS DA BUSCA")
        print(f"{'='*80}")
        print(f"📊 Combinações processadas:")
        print(f"   Total geradas: {stats['total_generated']:,}")
        print(f"   Conflitos ignorados: {stats['conflicts_skipped']:,}")
        print(f"   Testadas efetivamente: {combinations_tested:,}")
        
        print(f"\n🔍 Motivos de rejeição:")
        print(f"   Segmento muito pequeno: {stats['too_small']:,}")
        print(f"   Grupos insuficientes: {stats['insufficient_groups']:,}")
        print(f"   Uplift baixo: {stats['low_uplift']:,}")
        
        print(f"\n✅ Resultado:")
        print(f"   Segmentos válidos: {stats['valid_segments']:,}")
        valid_rate = (stats['valid_segments'] / combinations_tested * 100) if combinations_tested > 0 else 0
        print(f"   Taxa de sucesso: {valid_rate:.2f}%")
        
        
        print(f"\n{'='*80}")
        print(f"RESULTADO DA BUSCA")
        print(f"{'='*80}")
        print(f"Segmentos válidos encontrados: {len(best_segments)}")
        
        if len(best_segments) > 0:
            best_segments.sort(key=lambda x: x['additional_conversions'], reverse=True)
            
            best_uplift = best_segments[0]['uplift']
            best_additional = best_segments[0]['additional_conversions']
            
            print(f"Melhor uplift encontrado: {best_uplift:.2%}")
            print(f"Máximo de conversões adicionais: {best_additional:.1f}")
            
            if overall_stats['uplift'] != float('inf') and best_uplift > overall_stats['uplift']:
                improvement = best_uplift - overall_stats['uplift']
                print(f"🎯 Melhoria vs uplift geral: +{improvement:.2%}")
            
        return best_segments

    def evaluate_single_segment(self, df, conditions):
        """
        Avalia um segmento específico definido por condições
        
        Útil para testar segmentos específicos sem busca completa
        """
        segment = self.apply_conditions(df, conditions)
        
        if len(segment) < self.min_segment_size:
            return None
        
        test_count = len(segment[segment['group'] == 'test'])
        control_count = len(segment[segment['group'] == 'control'])
        
        if test_count < 30 or control_count < 30:
            return None
        
        uplift, test_conv, control_conv, is_sig, p_val = self.calculate_uplift_with_significance(segment)
        
        if uplift < self.min_uplift:
            return None
        
        additional_conversions = test_count * (test_conv - control_conv)
        
        return {
            'conditions': conditions,
            'segment_size': len(segment),
            'test_size': test_count,
            'control_size': control_count,
            'uplift': uplift,
            'test_conversion': test_conv,
            'control_conversion': control_conv,
            'additional_conversions': additional_conversions,
            'is_significant': is_sig,
            'p_value': p_val,
            'segment_data': segment
        }

    def print_results(self, segments, top_n=5):
        """
        Print results formatted
        """
        if not segments:
            print("Nenhum segmento encontrado com os critérios especificados.")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {min(top_n, len(segments))} SEGMENTOS COM MAIOR UPLIFT")
        print(f"{'='*80}")
        
        for i, segment in enumerate(segments[:top_n], 1):
            print(f"\nSegmento {i}:")
            print(f"  UPLIFT: {segment['uplift']:.2%}")
            print(f"  Conversão Teste: {segment['test_conversion']:.2%}")
            print(f"  Conversão Controle: {segment['control_conversion']:.2%}")
            print(f"  Conversões Adicionais: {segment['additional_conversions']:.1f}")
            
            if segment['is_significant']:
                print(f"  Estatisticamente Significativo (p={segment['p_value']:.4f})")
            else:
                print(f"  Não Significativo (p={segment['p_value']:.4f})")
            
            print(f"  Tamanho Total: {segment['segment_size']:,}")
            print(f"  Tamanho Teste: {segment['test_size']:,}")
            print(f"  Tamanho Controle: {segment['control_size']:,}")
            
            print(f"\n  Condições para este segmento:")
            for feature, op, value in segment['conditions']:
                if isinstance(value, float):
                    print(f"    • {feature} {op} {value:.2f}")
                else:
                    print(f"    • {feature} {op} {value}")

    def get_segment_data(self, df, conditions):
        """
        Returns data from a given segment
        """
        return self.apply_conditions(df, conditions)