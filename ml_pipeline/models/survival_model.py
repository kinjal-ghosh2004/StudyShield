from lifelines import CoxTimeVaryingFitter

class SurvivalAnalysisPredictor:
    def __init__(self, penalizer=0.1):
        self.penalizer = penalizer
        # We use a Time-Varying Cox Proportional Hazards model to handle data that changes every week
        self.model = CoxTimeVaryingFitter(penalizer=self.penalizer)
        
    def train(self, df_survival_format, id_col='id_student', start_col='start_time', stop_col='stop_time', event_col='event_occurred'):
        """
        Trains the Cox Time-Varying model on longitudinal event data.
        df_survival_format must contain columns for id, start, stop, and event flag.
        """
        print("Training Cox Time-Varying Survival Model...")
        self.model.fit(
            df_survival_format,
            id_col=id_col,
            event_col=event_col,
            start_col=start_col,
            stop_col=stop_col,
            show_progress=True
        )
        print("Training complete.")
        self.model.print_summary()
        
    def predict_hazard(self, df_current_state):
        """
        Predicts relative risk/hazard of dropout for currently active students based on their latest state.
        Higher hazard indicates higher immediate risk of event occurring.
        """
        return self.model.predict_partial_hazard(df_current_state)
        
    def plot_covariate_effects(self):
        """
        Visualizes the log-hazard ratio for each tracked behavioral or static factor.
        """
        self.model.plot()
