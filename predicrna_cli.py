#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PredicRNA CLI Tool
======================================
A command-line tool for predicting preeclampsia risk using cfRNA expression data.

Supports multiple models:
- Classification: Gaussian Naive Bayes (PE vs Normal)
- Severity: Random Forest (Mild/Moderate/Severe)
- Personalized Risk: Deep Learning (Time Series)

Usage:
    python predicrna_cli.py predict --input data.csv --output results.csv
    python predicrna_cli.py demo
    python predicrna_cli.py info
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== TensorFlow Import ====================

TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import load_model
    import keras
    TF_AVAILABLE = True
except ImportError:
    pass

# ==================== Configuration ====================

SCRIPT_DIR = Path(__file__).parent.absolute()

MODEL_PATHS = {
    'classification': SCRIPT_DIR / 'models' / 'best_model_gaussian_nb.pkl',
    'severity': SCRIPT_DIR / 'models' / 'best_model_random_forest.pkl',
    'personalized': SCRIPT_DIR / 'models' / 'Personalized_Risk_Model_best_model.keras', # Updated name
}

# Expected 44 features in specific order
EXPECTED_FEATURES = [
    'CCDC85B', 'CDC27', 'CDK6', 'DDX46', 'DDX6', 'GDI2', 'GP9', 'GSE1', 'HIPK1', 'IRAK3',
    'ISCA1', 'LMNA', 'LPIN2', 'MAP3K2', 'MTRNR2L8', 'Metazoa_SRP.161', 'NDUFA3', 'OAZ1',
    'OSBPL8', 'PABPN1', 'PPP1R12A', 'PSME4', 'RIOK3', 'RN7SL128P', 'RN7SL151P', 'RN7SL396P',
    'RN7SL4P', 'RN7SL5P', 'RN7SL674P', 'RN7SL752P', 'RN7SL767P', 'RNA5-8SP6', 'RNA5SP202',
    'RNA5SP352', 'RPGR', 'RPPH1', 'STAG2', 'STRADB', 'TAB2', 'TBL1XR1', 'U1.14', 'UBE2B',
    'USP34', 'YOD1'
]

MODEL_INFO = {
    'classification': {
        'name': 'Gaussian Naive Bayes',
        'task': 'Binary classification (Normal vs Preeclampsia)',
        'accuracy': '~85%',
        'type': 'sklearn'
    },
    'severity': {
        'name': 'Random Forest Classifier',
        'task': 'Multi-class (Mild/Moderate/Severe)',
        'accuracy': '~80%',
        'type': 'sklearn'
    },
    'personalized': {
        'name': 'Personalized Risk Model',
        'task': 'Individual risk assessment (Time Series)',
        'accuracy': '~79%',
        'type': 'keras',
        'inputs': ['cfrna_data (44)']
    }
}

# ==================== Model Manager ====================

class ModelManager:
    """Manage ML models for prediction"""
    
    def __init__(self):
        self.models = {}
        self.feature_names = EXPECTED_FEATURES
        
    def load_models(self, model_types=None):
        """Load specified models or all available models"""
        print("Loading models...")
        
        if model_types is None:
            model_types = ['classification', 'severity', 'personalized'] # Updated default models
        
        for model_name in model_types:
            if model_name not in MODEL_PATHS:
                continue
                
            model_path = MODEL_PATHS[model_name]
            model_info = MODEL_INFO.get(model_name, {})
            
            if not model_path.exists():
                print(f"  ⚠ {model_name}: Model file not found")
                continue
            
            try:
                if model_info.get('type') == 'keras':
                    if not TF_AVAILABLE:
                        print(f"  ⚠ {model_name}: TensorFlow not available")
                        continue
                    self.models[model_name] = load_model(str(model_path), compile=False)
                else:
                    self.models[model_name] = joblib.load(model_path)
                print(f"  ✓ Loaded {model_name} ({model_info.get('name', 'Unknown')})")
            except Exception as e:
                error_msg = str(e)[:80]
                print(f"  ✗ Failed to load {model_name}: {error_msg}")
        
        return len(self.models) > 0
    
    def get_model_info(self):
        """Get information about loaded models"""
        return {
            'available_models': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_details': {k: MODEL_INFO.get(k, {}) for k in self.models.keys()}
        }

# ==================== Data Processing ====================

def load_csv_data(filepath):
    """Load and validate CSV data"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def prepare_features(df, expected_features):
    """Prepare features in correct order"""
    id_column = None
    for col in ['patient_id', 'sample_id', 'id', 'ID']:
        if col in df.columns:
            id_column = col
            break
    
    if id_column:
        sample_ids = df[id_column].values
        df = df.drop(columns=[id_column])
    else:
        sample_ids = [f"Sample_{i+1}" for i in range(len(df))]
    
    feature_values = []
    missing_features = []
    
    for feature in expected_features:
        if feature in df.columns:
            feature_values.append(df[feature].values)
        else:
            found = False
            for col in df.columns:
                if col.strip() == feature or col.upper() == feature.upper():
                    feature_values.append(df[col].values)
                    found = True
                    break
            if not found:
                feature_values.append(np.zeros(len(df)))
                missing_features.append(feature)
    
    if missing_features:
        print(f"Warning: {len(missing_features)} features not found, filled with zeros")
        # Strict validation: If features are missing, exit.
        print(f"Error: Missing required features: {missing_features[:5]}...")
        sys.exit(1)
    
    X = np.column_stack(feature_values)
    return X, sample_ids

# ==================== Prediction Functions ====================

def predict_classification(model, X):
    """Perform classification prediction"""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        pe_prob = prob[1] if len(prob) > 1 else prob[0]
        normal_prob = prob[0] if len(prob) > 1 else 1 - prob[0]
        
        results.append({
            'prediction': 'Preeclampsia' if pred == 1 else 'Normal',
            'prediction_code': int(pred),
            'pe_probability': float(pe_prob),
            'normal_probability': float(normal_prob),
            'confidence': 'High' if max(pe_prob, normal_prob) > 0.8 else 'Medium' if max(pe_prob, normal_prob) > 0.6 else 'Low'
        })
    
    return results

def predict_severity(model, X):
    """Perform severity prediction"""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    severity_labels = ['Mild', 'Moderate', 'Severe']
    
    results = []
    for pred, prob in zip(predictions, probabilities):
        pred_idx = int(pred)
        severity = severity_labels[pred_idx] if pred_idx < len(severity_labels) else 'Unknown'
        
        results.append({
            'severity_level': severity,
            'severity_code': pred_idx,
            'severity_score': float(prob[pred_idx]) if pred_idx < len(prob) else 0.0,
        })
    
    return results

def predict_personalized(model, X):
    """Perform personalized prediction with Keras model"""
    if not TF_AVAILABLE:
        return None
    
    try:
        # Check model input shape
        input_shape = model.input_shape
        
        # Reshape X if necessary (add timestep dimension)
        if len(input_shape) == 3: # (None, Time, Features)
            X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))
        else:
            X_reshaped = X
            
        prediction = model.predict(X_reshaped, verbose=0)
        
        # Handle output
        if isinstance(prediction, list):
            prediction = prediction[-1]
            
        results = []
        for i in range(len(X)):
            # Flatten prediction
            prob = float(prediction[i].flatten()[0])
            prob = np.clip(prob, 0, 1)
            results.append({'probability': prob})
        
        return results
    except Exception as e:
        print(f"Error in personalized prediction: {e}")
        return None

def calculate_risk_score(pe_prob, severity_score=None):
    """Calculate overall risk score"""
    base_score = pe_prob * 100
    
    if severity_score:
        base_score = base_score * (1 + severity_score * 0.2)
    
    risk_score = min(base_score, 100)
    
    if risk_score >= 75:
        risk_level = 'Critical'
        recommendation = 'Immediate specialist consultation recommended'
    elif risk_score >= 50:
        risk_level = 'High'
        recommendation = 'Enhanced monitoring and preventive measures advised'
    elif risk_score >= 25:
        risk_level = 'Medium'
        recommendation = 'Regular monitoring of relevant indicators'
    else:
        risk_level = 'Low'
        recommendation = 'Continue routine prenatal care'
    
    return {
        'risk_score': round(risk_score, 2),
        'risk_level': risk_level,
        'recommendation': recommendation
    }

# ==================== Main Pipelines ====================

def run_prediction(input_file, output_file=None, model_manager=None):
    """Run the full prediction pipeline"""
    
    if model_manager is None:
        model_manager = ModelManager()
        if not model_manager.load_models(['classification', 'severity']):
            print("Error: No models available")
            return None
    
    df = load_csv_data(input_file)
    if df is None:
        return None
    
    X, sample_ids = prepare_features(df, model_manager.feature_names)
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    results = []
    
    for i, sample_id in enumerate(sample_ids):
        sample_X = X[i:i+1]
        result = {'sample_id': sample_id}
        
        if 'classification' in model_manager.models:
            cls_result = predict_classification(model_manager.models['classification'], sample_X)[0]
            result.update({
                'prediction': cls_result['prediction'],
                'pe_probability': cls_result['pe_probability'],
                'normal_probability': cls_result['normal_probability'],
                'confidence': cls_result['confidence'],
            })
        
        if 'severity' in model_manager.models:
            sev_result = predict_severity(model_manager.models['severity'], sample_X)[0]
            result.update({
                'severity_level': sev_result['severity_level'],
                'severity_score': sev_result['severity_score'],
            })
        
        if 'personalized' in model_manager.models:
            # Step 9 model
            pers_result = predict_personalized(model_manager.models['personalized'], sample_X)[0]
            result.update({
                'personalized_risk_score': pers_result['probability'] * 100
            })

        if 'pe_probability' in result:
            # Simple risk calculation for CLI output, aligning with web service Step 4.4 output logic if needed
            # Web service mainly outputs: Risk Prediction (Step 4.4), Time Series (Step 9), Severity (Step 4.5)
            # We keep 'risk_score' calculation here as a summary for CLI user
            risk = calculate_risk_score(result['pe_probability'], result.get('severity_score'))
            result.update({
                'risk_score': risk['risk_score'],
                'risk_level': risk['risk_level'],
                'recommendation': risk['recommendation'],
            })
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return results_df

def run_temporal_prediction(input_file, output_file=None, time_points=None, model_manager=None):
    """Run temporal multi-time-point prediction"""
    
    if time_points is None:
        time_points = [12, 16, 20, 24, 28, 32]
    
    if model_manager is None:
        model_manager = ModelManager()
        model_manager.load_models(['classification', 'personalized'])
    
    df = load_csv_data(input_file)
    if df is None:
        return None
    
    X, sample_ids = prepare_features(df, model_manager.feature_names)
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
    # Get base classification
    base_probs = []
    if 'classification' in model_manager.models:
        cls_model = model_manager.models['classification']
        for i in range(len(X)):
            prob = cls_model.predict_proba(X[i:i+1])[0]
            base_probs.append(prob[1] if len(prob) > 1 else prob[0])
    else:
        base_probs = [0.5] * len(X)
    
    # Check personalized model
    pers_model = model_manager.models.get('personalized')
    use_dl = pers_model is not None and TF_AVAILABLE
    
    if use_dl:
        print("Using Personalized Risk model for prediction")
    else:
        print("Using classification-based estimation (Model Step 9 missing)")
    
    results = []
    
    for i, sample_id in enumerate(sample_ids):
        sample_X = X[i:i+1]
        base_prob = base_probs[i]
        
        result = {
            'sample_id': sample_id,
            'base_pe_probability': round(base_prob, 4),
        }
        
        for week in time_points:
            if use_dl:
                # Use Step 9 model (Personalized Risk Model)
                try:
                    pred = predict_personalized(pers_model, sample_X)
                    if pred:
                        risk_prob = pred[0]['probability']
                    else:
                        risk_prob = base_prob
                except:
                    risk_prob = base_prob
            else:
                # Fallback to base probability
                risk_prob = base_prob

            risk_score = risk_prob * 100
            
            if risk_score >= 75:
                risk_level = 'Critical'
            elif risk_score >= 50:
                risk_level = 'High'
            elif risk_score >= 25:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            result[f'week{week}_probability'] = round(risk_prob, 4)
            result[f'week{week}_risk_score'] = round(risk_score, 2)
            result[f'week{week}_risk_level'] = risk_level
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return results_df

# ==================== CLI Commands ====================

def cmd_predict(args):
    """Handle predict command"""
    print("\n" + "="*60)
    print("cfRNA Preeclampsia Prediction")
    print("="*60 + "\n")
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    model_manager = ModelManager()
    model_manager.load_models(['classification', 'severity'])
    
    results = run_prediction(args.input, args.output, model_manager)
    
    if results is None:
        return 1
    
    print("\n" + "-"*60)
    print("PREDICTION RESULTS")
    print("-"*60)
    
    for _, row in results.iterrows():
        print(f"\n📋 Sample: {row['sample_id']}")
        print(f"   Prediction: {row.get('prediction', 'N/A')}")
        print(f"   PE Probability: {row.get('pe_probability', 0):.2%}")
        print(f"   Confidence: {row.get('confidence', 'N/A')}")
        if 'severity_level' in row:
            print(f"   Severity: {row['severity_level']}")
        if 'risk_level' in row:
            print(f"   Risk Level: {row['risk_level']} (Score: {row['risk_score']:.1f})")
        if 'recommendation' in row:
            print(f"   💡 {row['recommendation']}")
    
    print("\n" + "="*60)
    print(f"Total samples processed: {len(results)}")
    print("="*60 + "\n")
    
    return 0

def cmd_temporal(args):
    """Handle temporal prediction command"""
    print("\n" + "="*60)
    print("cfRNA Temporal Risk Prediction")
    print("="*60 + "\n")
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Parse time points
    if args.weeks:
        time_points = [int(w.strip()) for w in args.weeks.split(',')]
    else:
        time_points = [12, 16, 20, 24, 28, 32]
    
    results = run_temporal_prediction(args.input, args.output, time_points)
    
    if results is None:
        return 1
    
    print("\n" + "-"*60)
    print("TEMPORAL RISK PREDICTIONS")
    print("-"*60)
    
    for _, row in results.iterrows():
        print(f"\n📋 Sample: {row['sample_id']}")
        print(f"   Base PE Probability: {row['base_pe_probability']:.2%}")
        print(f"\n   Risk by Gestational Week:")
        
        for week in time_points:
            prob_col = f'week{week}_probability'
            score_col = f'week{week}_risk_score'
            level_col = f'week{week}_risk_level'
            
            if prob_col in row and pd.notna(row[prob_col]):
                level = row[level_col]
                emoji = '🔴' if level == 'Critical' else '🟠' if level == 'High' else '🟡' if level == 'Medium' else '🟢'
                print(f"      Week {week:2d}: {emoji} {row[score_col]:5.1f}% ({level})")
    
    print("\n" + "="*60)
    print(f"Total samples processed: {len(results)}")
    print("="*60 + "\n")
    
    return 0

def cmd_demo(args):
    """Handle demo command"""
    print("\n" + "="*60)
    print("cfRNA Prediction Demo")
    print("="*60 + "\n")
    
    example_data_path = SCRIPT_DIR / 'examples' / 'sample_data.csv'
    
    if not example_data_path.exists():
        print("Error: Demo data file not found.")
        return 1
    
    output_file = SCRIPT_DIR / 'examples' / 'demo_results.csv'
    
    print(f"Running demo with sample data: {example_data_path}\n")
    
    model_manager = ModelManager()
    model_manager.load_models(['classification', 'severity', 'temporal'])
    
    results = run_prediction(str(example_data_path), str(output_file), model_manager)
    
    if results is None:
        return 1
    
    print("\n" + "-"*60)
    print("DEMO RESULTS SUMMARY")
    print("-"*60)
    
    for _, row in results.iterrows():
        print(f"\n📋 Sample: {row['sample_id']}")
        print(f"   🔬 Prediction: {row.get('prediction', 'N/A')}")
        print(f"   📊 PE Probability: {row.get('pe_probability', 0):.2%}")
        print(f"   ⚡ Risk Level: {row.get('risk_level', 'N/A')}")
        print(f"   💡 {row.get('recommendation', 'N/A')}")
    
    # Run temporal demo too
    print("\n" + "-"*60)
    print("TEMPORAL PREDICTION DEMO (Weeks: 12, 20, 28)")
    print("-"*60)
    
    temporal_output = SCRIPT_DIR / 'examples' / 'demo_temporal_results.csv'
    temporal_results = run_temporal_prediction(str(example_data_path), str(temporal_output), [12, 20, 28])
    
    if temporal_results is not None:
        for _, row in temporal_results.iterrows():
            print(f"\n📋 Sample: {row['sample_id']}")
            for week in [12, 20, 28]:
                level = row[f'week{week}_risk_level']
                score = row[f'week{week}_risk_score']
                emoji = '🔴' if level == 'Critical' else '🟠' if level == 'High' else '🟡' if level == 'Medium' else '🟢'
                print(f"   Week {week:2d}: {emoji} {score:5.1f}% ({level})")
    
    print("\n" + "="*60)
    print(f"Demo completed! Results saved to: {output_file}")
    print("="*60 + "\n")
    
    return 0

def cmd_info(args):
    """Handle info command"""
    print("\n" + "="*60)
    print("cfRNA Prediction System Information")
    print("="*60 + "\n")
    
    model_manager = ModelManager()
    model_manager.load_models()
    
    info = model_manager.get_model_info()
    
    print("📌 Available Models:")
    for model in info['available_models']:
        details = MODEL_INFO.get(model, {})
        print(f"   • {model}")
        print(f"     Name: {details.get('name', 'Unknown')}")
        print(f"     Task: {details.get('task', 'Unknown')}")
        print(f"     Accuracy: {details.get('accuracy', 'Unknown')}")
    
    print(f"\n📌 TensorFlow Available: {'Yes' if TF_AVAILABLE else 'No'}")
    
    print(f"\n📌 Required Features: {info['feature_count']} cfRNA markers")
    print("\n📌 Feature List:")
    for i, feature in enumerate(info['feature_names'], 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n📌 Usage Examples:")
    print("   # Basic prediction")
    print("   python cfrna_cli.py predict -i data.csv -o results.csv")
    print("\n   # Temporal multi-week prediction")
    print("   python cfrna_cli.py temporal -i data.csv --weeks 12,20,28")
    print("\n   # Run demo")
    print("   python cfrna_cli.py demo")
    
    print("\n" + "="*60 + "\n")
    
    return 0

def cmd_validate(args):
    """Handle validate command"""
    print("\n" + "="*60)
    print("CSV Format Validation")
    print("="*60 + "\n")
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        return 1
    
    try:
        df = pd.read_csv(args.input)
        print(f"✓ CSV file loaded successfully")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        found = sum(1 for f in EXPECTED_FEATURES if f in df.columns)
        missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
        
        print(f"\n📊 Feature Check:")
        print(f"   Found: {found}/{len(EXPECTED_FEATURES)}")
        print(f"   Missing: {len(missing)}")
        
        if missing and len(missing) <= 10:
            print(f"\n⚠️  Missing features:")
            for f in missing:
                print(f"      - {f}")
        
        print("\n" + "="*60 + "\n")
        return 0
        
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return 1

# ==================== Main Entry Point ====================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='cfRNA Preeclampsia Prediction CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s predict -i data.csv -o results.csv
  %(prog)s temporal -i data.csv --weeks 12,20,28
  %(prog)s demo
  %(prog)s info

For more information, visit: https://github.com/your-repo/cfrna-cli
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction on input data')
    predict_parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    predict_parser.add_argument('--output', '-o', help='Output CSV file path')
    
    # temporal command
    temporal_parser = subparsers.add_parser('temporal', help='Run temporal multi-week prediction')
    temporal_parser.add_argument('--input', '-i', required=True, help='Input CSV file path')
    temporal_parser.add_argument('--output', '-o', help='Output CSV file path')
    temporal_parser.add_argument('--weeks', '-w', default='12,16,20,24,28,32',
                                help='Gestational weeks (comma-separated)')
    
    # demo command
    subparsers.add_parser('demo', help='Run demo with sample data')
    
    # info command
    subparsers.add_parser('info', help='Show system information')
    
    # validate command
    validate_parser = subparsers.add_parser('validate', help='Validate CSV format')
    validate_parser.add_argument('--input', '-i', required=True, help='CSV file to validate')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        'predict': cmd_predict,
        'temporal': cmd_temporal,
        'demo': cmd_demo,
        'info': cmd_info,
        'validate': cmd_validate,
    }
    
    return commands.get(args.command, lambda _: parser.print_help() or 1)(args)

if __name__ == '__main__':
    sys.exit(main())
