class UnifiedModelSearch:
    """
    Search across different models AND their parameters in a single GridSearch.
    """
    
    def __init__(self, random_state=42, n_jobs=-1, cv_splits=5, scoring='balanced_accuracy', memory=None):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.memory = memory
        self.grid_search = None
        self.results = None
        
    def create_unified_search(self):
        """Create a single GridSearch that searches across all models."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression()),  # Placeholder
        ], memory=self.memory)
        
        # Define all models and their specific parameters
        param_grid = [
            {
                'classifier': [LogisticRegression(tol=1e-3, dual=False, random_state=self.random_state, max_iter=1000)],
                'classifier__C': np.logspace(-2, 2, num=5),
                'classifier__class_weight': [None, 'balanced'],
                'classifier__penalty': ['none', 'l2'],
            },
            {
                'classifier': [RidgeClassifier(tol=1e-3, random_state=self.random_state, max_iter=1000)],
                'classifier__alpha': np.logspace(-2, 2, num=5),
                'classifier__class_weight': [None, 'balanced'],
            },
            {
                'classifier': [LinearSVC(tol=1e-3, random_state=self.random_state, max_iter=1000)],
                'classifier__C': np.logspace(-2, 2, num=5),
                'classifier__class_weight': [None, 'balanced'],
                'classifier__penalty': ['l2'],
                'classifier__loss': ['hinge', 'squared_hinge'],
            },
            {
                'classifier': [SVC(tol=1e-3, random_state=self.random_state, max_iter=1000)],
                'classifier__C': np.logspace(-2, 2, num=5),
                'classifier__class_weight': [None, 'balanced'],
                'classifier__gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001],
                'classifier__kernel': ['linear', 'rbf'],
            }
        ]
        
        self.grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
            cv=StratifiedKFold(n_splits=self.cv_splits),
            verbose=2,
        )
        
        return self.grid_search
    
    def fit(self, X_train, y_train):
        """Fit the unified GridSearch."""
        if self.grid_search is None:
            self.create_unified_search()
        
        print("Fitting unified GridSearch across all models...")
        self.grid_search.fit(X_train, y_train)
        
        self.results = {
            'best_score': self.grid_search.best_score_,
            'best_params': self.grid_search.best_params_,
            'best_estimator': self.grid_search.best_estimator_,
            'cv_results': pd.DataFrame(self.grid_search.cv_results_)
        }
        
        return self.results
    
    def print_results(self):
        """Print the results."""
        if self.results is None:
            print("No results available. Run fit() first.")
            return
        
        print("\n" + "="*80)
        print("UNIFIED GRID SEARCH RESULTS")
        print("="*80)
        print(f"Best CV Score: {self.results['best_score']:.4f}")
        print(f"\nBest Parameters:")
        for param, value in self.results['best_params'].items():
            print(f"  {param}: {value}")
