# 🎬 Hybrid Movie Recommendation System

A sophisticated movie recommendation system that combines content-based and collaborative filtering approaches to provide personalized movie suggestions. Built with Python and Streamlit for an interactive web interface.

## 🚀 Features

- **Hybrid Recommendation Engine**: Combines content-based and collaborative filtering for better accuracy
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Multiple Recommendation Types**: 
  - Content-Based: Recommendations based on movie genres and features
  - Collaborative Filtering: Recommendations based on user behavior patterns
  - Hybrid: Combined approach for enhanced accuracy
- **Customizable Results**: Adjustable number of recommendations (1-10)
- **User-Specific Recommendations**: Personalized suggestions based on user ID
- **Model Persistence**: Save and load trained models for production use

## 🏗️ Project Structure

```
Recommendation-app/
├── app.py                          # Main Streamlit application
├── pyproject.toml                  # Project configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py           # Data loading utilities
│   └── models/
│       ├── __init__.py
│       └── recommender.py         # Hybrid recommendation engine
└── hybrid_movie_recommender.egg-info/
    ├── dependency_links.txt
    ├── PKG-INFO
    ├── requires.txt
    ├── SOURCES.txt
    └── top_level.txt
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms (TF-IDF, Cosine Similarity, KNN)
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Pickle**: Model serialization

### Key Libraries:
- `sklearn.metrics.pairwise.cosine_similarity`: Content-based similarity calculation
- `sklearn.feature_extraction.text.TfidfVectorizer`: Text feature extraction
- `sklearn.neighbors.NearestNeighbors`: Collaborative filtering
- `streamlit`: Interactive web interface

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Movie dataset in the required format (movies.dat, ratings.dat)

## 🔧 Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd ML_PROJECTS/Recommendation-app
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

## 🚀 Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Use the interface**:
   - Select a User ID from the dropdown
   - Choose a movie you like from the movie list
   - Set the number of recommendations (1-10)
   - Select recommendation type (Hybrid, Content-Based, or Collaborative)
   - Click "Get Recommendations" to see results

### Using the Recommendation Engine Programmatically

```python
from src.models.recommender import HybridRecommender
from src.data.load_data import movies, ratings

# Initialize the recommender
recommender = HybridRecommender(movies, ratings)

# Train the model
recommender.train()

# Get recommendations
recommendations = recommender.recommend(
    user_id=1,
    title="Toy Story (1995)",
    top_n=5
)

# Access different types of recommendations
print("Content-based:", recommendations['content'])
print("Collaborative:", recommendations['collaborative'])
print("Hybrid:", recommendations['hybrid'])
```

## 🧠 How It Works

### 1. Content-Based Filtering
- Uses TF-IDF vectorization on movie genres
- Calculates cosine similarity between movies
- Recommends movies similar to the selected movie based on genre features

### 2. Collaborative Filtering
- Creates a user-item matrix from ratings data
- Uses K-Nearest Neighbors to find similar movies based on user behavior
- Recommends movies that similar users have rated highly

### 3. Hybrid Approach
- Combines both content-based and collaborative recommendations
- Aggregates results from both methods
- Provides more robust and diverse recommendations

- **User-Specific Recommendations**: Personalized suggestions based on user ID
- **Model Persistence**: Save and load trained models for production use

## 📊 Performance Metrics

We benchmark the system latency and recommendation quality using `benchmark.py`.

| Metric | Value (Approx.) | Description |
| :--- | :--- | :--- |
| **Training Time** | < 0.1s | Fast model retraining on small datasets |
| **Query Latency** | ~3ms | Ultra-low latency for real-time inference |
| **Precision@10** | *Dataset Dependent* | Measures relevance of top 10 recommendations |

*Note: Latency metrics were measured on a local environment using synthetic benchmark data.*

## 📊 Data Format

The system expects movie and rating data in the following format:

**Movies Data (movies.dat)**:
```
movieId::title::genres
1::Toy Story (1995)::Animation|Children's|Comedy
```

**Ratings Data (ratings.dat)**:
```
userId::movieId::rating::timestamp
1::1::5::978300760
```

## 🔧 Configuration

### Model Parameters
- **TF-IDF Vectorizer**: English stop words removal
- **Cosine Similarity**: Content-based similarity metric
- **KNN**: Cosine distance for collaborative filtering
- **Default Recommendations**: 5 movies per request

### Customization Options
- Modify similarity metrics in `recommender.py`
- Adjust TF-IDF parameters for content analysis
- Change KNN parameters for collaborative filtering
- Customize UI components in `app.py`

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
1. **Docker** (recommended):
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Cloud Platforms**:
   - Streamlit Cloud
   - Heroku
   - AWS/GCP/Azure

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📈 Performance Considerations

- **Model Training**: Train once, use multiple times
- **Memory Usage**: Large datasets may require optimization
- **Response Time**: Consider caching for frequently requested recommendations
- **Scalability**: Use model persistence for production environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is part of the ML_PROJECTS repository. Please refer to the main repository for license information.

## 👤 Author

**Pawan Parida**
- Email: 13zero7two005@gmail.com
- GitHub: [@zer-art](https://github.com/zer-art)

## 🔮 Future Enhancements

- [ ] Deep learning-based recommendations
- [ ] Real-time model updates
- [ ] A/B testing framework
- [ ] Advanced evaluation metrics
- [ ] Support for additional data sources (IMDb, TMDb)
- [ ] User feedback integration
- [ ] Recommendation explanations
- [ ] Multi-language support

## 📚 References

- [Recommender Systems: The Textbook](https://www.springer.com/gp/book/9783319296579)
- [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [Content-Based Filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

⭐ If you found this project helpful, please give it a star!