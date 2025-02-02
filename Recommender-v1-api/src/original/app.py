from flask import Flask, request, jsonify
from V2 import optimize_model_parameters, generate_results, load_user_artists
from pathlib import Path
import csv
import os

app = Flask(__name__)

if not os.path.exists('results'):
    os.makedirs('results')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        user_artists = load_user_artists(Path("../../dataset/user_artists.dat"))
        factors, regularization = optimize_model_parameters(user_artists)
        
        with open('results/optimized_params.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['factors', 'regularization'])
            writer.writerow([factors, regularization])
        
        return jsonify({
            "message": "Optimization complete",
            "factors": factors,
            "regularization": regularization
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_index = int(request.args.get('user_index', 2))
        recommend_limit = int(request.args.get('limit', 10))
        
        generate_results(user_index=user_index, recommend_limit=recommend_limit)
        
        return jsonify({
            "message": f"Recommendations generated for user {user_index}",
            "results_file": f"results/result_user_{user_index}.csv",
            "image_file": f"results/result_user_{user_index}.png"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)