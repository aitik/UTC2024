class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.last_ga_run_day = 0  # Track the last day GA was run
        # Initial setup
        self.train_data = train_data.copy()
        self.day_counter = 0
        self.ga_run_frequency = 3  # Example: Trigger GA every 15 days
        self.pop_size = 25
        self.mutation_rate = 0.15

        # Initialize GA components
        self.high_fitness_weights = {}
        self.convergence_count = []
        self.initialize_ga()

    def initialize_ga(self):
        """Initial GA run to set up high fitness weights."""
        proj_dfs = self.gen_data(self.train_data, 0, 5, 30)
        gen1 = self.gen_population(self.pop_size, self.train_data)
        sharpes = self.evaluate_population(gen1, proj_dfs)
        self.genetic_algo(sharpes, gen1, self.pop_size, self.mutation_rate, self.train_data, proj_dfs)

    def daily_update(self, asset_prices):
        """Updates weights based on GA, triggered periodically."""
        self.day_counter += 1
        if self.day_counter >= self.ga_run_frequency:
            # Time to run one GA generation
            proj_dfs = self.gen_data(self.train_data, 0, 5, 5)
            current_population = self.gen_population(self.pop_size, self.train_data)
            sharpes = self.evaluate_population(current_population, proj_dfs)
            # Update high_fitness_weights based on the new generation
            self.genetic_algo(sharpes, current_population, self.pop_size, self.mutation_rate, self.train_data, proj_dfs)
            self.day_counter = 0  # Reset counter

        # Allocate based on the latest high_fitness_weights
        max_sharpe = max(self.high_fitness_weights.keys())
        best_weights = self.high_fitness_weights[max_sharpe]
        return np.clip(best_weights, -1, 1)

    def update_ga(self, data):
        """Runs the GA to update portfolio weights."""
        proj_dfs = self.gen_data(data, first_day=0, runs=5, days=30)
        pop_size = 25
        mutation_rate = 0.15
        
        gen1 = self.gen_population(pop_size, data)
        sharpes = self.evaluate_population(gen1, proj_dfs)
        self.genetic_algo(sharpes, gen1, pop_size, mutation_rate, data, proj_dfs)
        
        max_sharpe = max(self.high_fitness_weights.keys())
        self.best_weights = self.high_fitness_weights[max_sharpe]

    def allocate_portfolio(self, asset_prices):
        """Dynamically computes weights for portfolio allocation."""
        current_day = len(self.running_price_paths)
        self.running_price_paths = self.running_price_paths._append(pd.DataFrame([asset_prices], columns=self.running_price_paths.columns), ignore_index=True)
        # Check if GA should run again based on frequency
        if current_day - self.last_ga_run_day >= self.ga_run_frequency:
            self.update_ga(self.running_price_paths)
            self.last_ga_run_day = current_day
        
        # Here, you could adjust self.best_weights based on recent data before the next GA run
        # For simplicity, this step is omitted but consider incorporating market trends or volatility
        
        # Append the new prices to running_price_paths for future GA runs
        
        return np.clip(self.best_weights, -1, 1)
    # ... All your genetic algorithm functions go here ...
    # make_df, gen_data, gen_weights, gen_population, evaluate_chrom,
    # evaluate_population, choose_chrom, crossover, mutate, rebalance, next_gen, genetic_algo
    def make_df(self, df):
        df['shift']=df[df.columns[0]].shift(1)
        df['PctChg']=(df[df.columns[0]]-df["shift"]).div(df["shift"])*100
        df=df.drop(columns='shift')
        return df
    # Generate list of projections
    def gen_data(self, df, first_day, runs, days):
        
        # Create list of arrays with projected daily returns
        data=[]
        first_prices=[]
        for x in range(len(df.columns)):
            temp=self.make_df(df.iloc[:,x:x+1])
            first_prices.append(temp[temp.columns[0]].iloc[first_day-1])
            data.append(np.random.laplace(loc=temp['PctChg'].mean(), scale=temp['PctChg'].std(), size=(runs, days)))
        
        # Get projected prices by applying % changes
        proj_dfs=[]
        for asset_num in range(len(data)):
            
            asset_data=[]
            for n in range(len(data[asset_num])):
                prices=[first_prices[asset_num]]
                for x in range(len(data[asset_num][n])):
                    prices.append(prices[x]*(100+data[asset_num][n][x])/100)
                asset_data.append(prices)
            
            proj_arr=np.array(asset_data)
            proj_df=pd.DataFrame()
            for x in range(proj_arr.shape[0]):
                proj_df[str(x)]=proj_arr[x]
            proj_dfs.append(proj_df)
            
        # Transform dataframes
        dfs=[]
        for run_num in range(proj_dfs[0].shape[1]):
            temp=pd.DataFrame()
            for asset_num in range(len(proj_dfs)):
                prev_df=proj_dfs[asset_num]
                temp[str(asset_num)]=prev_df[prev_df.columns[run_num]]
            dfs.append(temp)
            
        return dfs
    def gen_weights(self, df):
        weights=np.random.randint(-100000, 100000, len(df.columns))
        weights=weights/sum(weights)
        return weights
    def gen_population(self, num_chrom, df):
        population=[]
        for x in range(num_chrom):
            population.append(self.gen_weights(df))
        return np.array(population)
    def evaluate_chrom(self, chromosome, dfs):    
        sharpes=[]
        for x in range(len(dfs)):
            temp=dfs[x].multiply(chromosome, axis=1)
            test=temp[list(temp.columns)].sum(axis=1).pct_change()
            sharpe=(test.mean()/test.std())*year_num
            sharpes.append(sharpe)
        return np.mean(sharpes)
    def evaluate_population(self, pop, proj_dfs):
        sharpes=[]
        for chrom in pop:
            sharpes.append(self.evaluate_chrom(chrom, proj_dfs))
        return np.array(sharpes)
    # Choose chromosome for next generation given the cumulataive normalized sharpes
    def choose_chrom(self, population, cum_norm_sharpes):
        for n in range(len(cum_norm_sharpes)):
            if cum_norm_sharpes[n]>np.random.rand(1)[0]:
                return population[n]
    def crossover(self, chrom1, chrom2):
        if np.random.rand(1)[0]>.5:
            return np.concatenate((chrom1[:int(len(chrom1)/2)], chrom2[int(len(chrom1)/2):]), axis=None)
        else:
            return np.concatenate((chrom2[:int(len(chrom1)/2)], chrom1[int(len(chrom1)/2):]), axis=None)
    def mutate(self, chrom, rate):
        new=[]
        for weight in chrom:
            if np.random.rand(1)[0]<rate:
                new_weight=weight*(1+np.random.normal(0, .4, 1)[0])
                if(new_weight<0):
                    new.append(0)
                else:
                    new.append(new_weight)
            else:
                new.append(weight)
        return np.array(new)
    def rebalance(self, chrom):
        return chrom/sum(chrom)
    # Create next generation of chromosomes (weights)
    def next_gen(self, sharpes, population, mutation_rate):
        
        new_gen=[]
        
        # Select best fourth
        num_chosen_direct=round(len(population)/4)
        temp={}
        for x in range(len(sharpes)):
            temp[x]=sharpes[x]
        temp={k: v for k, v in sorted(temp.items(), key=lambda item: item[1])}
        keys=list(temp.keys())[-1*num_chosen_direct:]
        for x in keys:
            new_gen.append(population[x])
        
        # Select rest through crossover: create cumulative norm fitness list
        norm_sharpes=sharpes/sum(sharpes)
        cum_norm_sharpes=[norm_sharpes[0]]
        for n in range(1, len(norm_sharpes)):
            cum_norm_sharpes.append(cum_norm_sharpes[n-1]+norm_sharpes[n])
        for x in range(len(population)-num_chosen_direct):
            new_gen.append(self.crossover(self.choose_chrom(population, cum_norm_sharpes), self.choose_chrom(population, cum_norm_sharpes)))
            
        # Mutation and rebalance
        final=[]
        for x in new_gen:
            final.append(self.rebalance(self.mutate(x, mutation_rate)))
            
        return np.array(final)
    def genetic_algo(self, prev_gen_sharpes, prev_gen, pop_size, mutation_rate, df, proj_dfs):
        
        # Add to high fitness weights dict
        max_sharpe=max(prev_gen_sharpes)
        best_weights=prev_gen[list(prev_gen_sharpes).index(max_sharpe)]
        self.high_fitness_weights[max_sharpe]=best_weights
        
        # Check convergence
        convergence=False
        if (len(self.high_fitness_weights)>=20):
            convergence=True
        elif (len(self.high_fitness_weights)>1):
            if max_sharpe<list(self.high_fitness_weights.keys())[-2]*1.02:
                self.convergence_count.append(1)
            else:
                self.convergence_count.append(0)

            if (sum(self.convergence_count[-20:])==20):
                convergence=True
            else:
                convergence=False
        else:
            self.convergence_count.append(0)
        
        # Recursive GA
        if (convergence==False):
            print("Generation Number "+str(len(self.convergence_count)+1))
            print("---Processing")
            print("---Sharpe: "+str(max_sharpe))
            new_gen=self.next_gen(prev_gen_sharpes, prev_gen, mutation_rate)
            new_gen_sharpes=self.evaluate_population(new_gen, proj_dfs)
            print("---Done")
            self.genetic_algo(new_gen_sharpes, new_gen, pop_size, mutation_rate, df, proj_dfs)
        else:
            print("Convergence achieved")

# 0.11 (80/20)
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


data = pd.read_csv('case_2/Case 2 Data 2024.csv', index_col = 0)

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.2, shuffle = False)

import numpy as np
from scipy.optimize import minimize

class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()

        # Parameters (adjust as needed)
        self.window_size = 200  # Lookback window for rolling calculations
        self.momentum_window = 40  # Window for momentum calculation
        self.momentum_weight = 1  # Weight assigned to momentum in hybrid strategy
        self.normalized_momentum = np.zeros(len(train_data.columns))

    def allocate_portfolio(self, asset_prices):
        self.running_price_paths = pd.concat([self.running_price_paths, pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)], ignore_index=True)
        # return np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]) # Compare to naive implementation
        # Calculate returns using a rolling window
        if len(self.running_price_paths) > self.window_size:
            returns = self.running_price_paths.tail(self.window_size).pct_change().dropna()
        else:
            returns = self.running_price_paths.pct_change().dropna()

        # Calculate momentum only once per momentum window
        if len(self.running_price_paths) % self.momentum_window == 0:
            momentum = self.running_price_paths.tail(self.momentum_window).pct_change().dropna().sum(axis=0)
            self.normalized_momentum = (momentum - momentum.min()) / (momentum.max() - momentum.min())

        # Define objective function (negative Sharpe Ratio)
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(returns.mean() * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
            if self.normalized_momentum is None:
                return 0
            # Use stored momentum values
            return -(portfolio_return + self.momentum_weight * np.sum(self.normalized_momentum * weights)) / portfolio_std

        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x + 1},
                       {'type': 'ineq', 'fun': lambda x: 1 - x})

        # Initial guess for weights (equal allocation)
        initial_weights = np.ones(len(asset_prices)) / len(asset_prices)

        # Optimize portfolio weights
        result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', constraints=constraints)
        weights = result.x
        weights = np.clip(weights, -1, 1)
        return weights


## Hedge algorithm (pretty bad)
class Allocator():
    def __init__(self, train_data):
        self.num_assets = train_data.shape[1]
        self.weights = np.zeros(self.num_assets)  # Start with neutral weights
        self.eta = -0.01  # Initial learning rate
        self.performance_history = []  # Track performance to adjust learning rate
        
        # Calculate returns from price data
        self.returns_history = train_data.pct_change().dropna()
        self.train_on_initialization()

    def train_on_initialization(self):
        # Use historical data to set initial weights
        for day_returns in self.returns_history.values:
            self.update_weights(day_returns)  # Directly use returns as "loss"

    def allocate_portfolio(self, asset_prices):
        # Calculate returns for the current day
        last_prices = (self.returns_history.iloc[-1] + 1).values
        today_returns = asset_prices / last_prices - 1
        
        # Update weights based on today's returns
        self.update_weights(today_returns)
        
        # Record performance (simple return for demonstration)
        self.performance_history.append(np.dot(self.weights, today_returns))
        
        # Adjust learning rate based on recent performance
        self.adjust_learning_rate()
        
        # Append today's returns for future allocations
        self.returns_history = self.returns_history._append(pd.Series(today_returns, index=self.returns_history.columns), ignore_index=True)
        
        return self.weights

    def update_weights(self, returns):
        # Update weights proportional to returns
        self.weights += self.eta * returns
        # Ensure weights are within the [-1, 1] range and sum up to 1
        self.weights = np.clip(self.weights, -1, 1)
        self.weights /= np.sum(np.abs(self.weights))

    def adjust_learning_rate(self):
        # Simple performance-based adjustment for demonstration
        recent_performance = self.performance_history[-5:]  # Last 5 periods
        if len(recent_performance) > 1:
            # Reduce learning rate if performance has been positive, indicating confidence
            if np.mean(recent_performance) > 0:
                self.eta *= 0.9
            else:
                # Increase learning rate to explore more when recent performance is negative
                self.eta *= 1.1
            self.eta = np.clip(self.eta, 0.001, 0.05)

## Keras Deep Learning method
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('case_2/Case 2 Data 2024.csv', index_col=0)
TRAIN, TEST = train_test_split(data, test_size=0.2, shuffle=False)

class Allocator():
    def __init__(self, train_data):
        # Data preprocessing
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(train_data)
        
        # Create sequences for training
        self.X_train, self.y_train = self.create_sequences(scaled_data, n_past=60)

        # Build and compile the LSTM model
        self.model = self.build_model(input_shape=(self.X_train.shape[1], self.X_train.shape[2]))

        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=0)  # Set verbose to 1 if you want to see the training output

        self.running_price_paths = train_data.copy()

    def create_sequences(self, data, n_past):
        X, y = [], []
        for i in range(n_past, len(data)):
            X.append(data[i-n_past:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(input_shape[1]))  # output layer with input_shape[1] units (same as the number of assets)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def allocate_portfolio(self, asset_prices):
        # Prepare the data for prediction
        last_window = self.running_price_paths[-60:].values  # Assuming n_past is 60
        last_window_scaled = self.scaler.transform(last_window)
        last_window_scaled = last_window_scaled.reshape((1, last_window_scaled.shape[0], last_window_scaled.shape[1]))
        
        # Predict the returns for the next day
        predicted_returns = self.model.predict(last_window_scaled)[0]
        
        # Determine weights based on predicted returns
        # Apply a transformation and normalization to ensure the weights sum up to 1 and are within [-1, 1]
        weights = self.transform_returns_to_weights(predicted_returns)
        
        # Update running price paths with the new prices
        new_data = pd.DataFrame([asset_prices], columns=self.running_price_paths.columns)
        self.running_price_paths = self.running_price_paths._append(new_data, ignore_index=True)
        
        return weights

    def transform_returns_to_weights(self, returns):
        # Apply a transformation to ensure the weights sum up to 1 and are within [-1, 1]
        weights = np.tanh(returns)  # Use tanh to ensure the weights are between [-1, 1]
        weights /= np.sum(np.abs(weights))  # Normalize so they sum up to 1
        return weights
    

## New winner (0.1103)
class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.values  # Using numpy array for speed
        self.columns = train_data.columns
        self.window_size = 95
        self.normalized_momentum = None
        self.update_momentum()  # Initialize momentum calculation

    def update_momentum(self):
        if self.running_price_paths.shape[0] >= self.window_size:
            momentum = pd.DataFrame(self.running_price_paths).pct_change().dropna().sum(axis=0)
            normalized_momentum = (momentum - momentum.min()) / (momentum.max() - momentum.min())
            self.normalized_momentum = normalized_momentum.values  # Convert to numpy array
    
    def allocate_portfolio(self, asset_prices):
        # Append new prices
        self.running_price_paths = np.vstack([self.running_price_paths, asset_prices])
        self.update_momentum()
        # return np.array([1/6,1/6,1/6,1/6,1/6,1/6]) # Compare to naive 1/n method
        # Calculate rolling returns
        returns = np.diff(self.running_price_paths[-self.window_size:], axis=0) / self.running_price_paths[-self.window_size:-1]
        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns.T)

        # Objective function for MVO: minimize negative Sharpe ratio
        def objective_function(weights, mean_returns, cov_matrix):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility

        # Constraints and bounds for MVO
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights must sum to 1
        bounds = [(-1, 1) for _ in range(len(asset_prices))]  # Weights are bounded between -1 and 1

        # Initial guess for the optimizer
        initial_guess = np.ones(len(asset_prices)) / len(asset_prices)

        # Perform optimization to get the optimal weights
        result = scipy.optimize.minimize(
            objective_function, initial_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        # The optimal weights
        weights = result.x
        # weights = np.clip(weights, -1, 1)
        # weights = weights/np.sqrt(weights.dot(weights))

        return weights
    def set_window_size(self, window_size):
        self.window_size = window_size
        self.update_momentum()


##V2
class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.values  # Using numpy array for speed
        self.window_size = 95
        self.normalized_momentum = None
        self.update_momentum()  # Initialize momentum calculation

    def update_momentum(self):
        if self.running_price_paths.shape[0] >= self.window_size:
            momentum = pd.DataFrame(self.running_price_paths).pct_change().dropna().sum(axis=0)
            normalized_momentum = (momentum - momentum.mean()) / momentum.std()
            self.normalized_momentum = normalized_momentum.values  # Convert to numpy array
    
    def allocate_portfolio(self, asset_prices):
        # Append new prices
        self.running_price_paths = np.vstack([self.running_price_paths, asset_prices])
        self.update_momentum()
        # return np.array([1/6,1/6,1/6,1/6,1/6,1/6]) # Compare to naive 1/n method
        # return np.array([0.3121, -0.1765, 0.0313, 0.0574, 0.1671, 0.2556]) # Compare to some good static method
        # Calculate rolling returns
        returns = np.diff(self.running_price_paths[-self.window_size:], axis=0) / self.running_price_paths[-self.window_size:-1]
        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns.T)

        # Objective function for MVO: minimize negative Sharpe ratio
        def objective_function(weights, mean_returns, cov_matrix):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility

        # Constraints and bounds for MVO
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights must sum to 1
        bounds = [(-1, 1) for _ in range(len(asset_prices))]  # Weights are bounded between -1 and 1

        # Initial guess for the optimizer
        initial_guess = self.normalized_momentum / np.sum(np.abs(self.normalized_momentum))

        # Perform optimization to get the optimal weights
        result = scipy.optimize.minimize(
            objective_function, initial_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        # The optimal weights
        weights = result.x
        weights = np.clip(weights, -1, 1)

        total_weight_sum = np.sum(weights)
        if total_weight_sum != 0:
            weights = weights / (total_weight_sum * 1.01)
        else:
            weights = np.full(weights.shape, 1 / weights.size)
        return weights
    def set_window_size(self, window_size):
        self.window_size = window_size
        self.update_momentum()


## Pair Trading
class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        
        self.running_price_paths = train_data.copy()
        
        self.train_data = train_data.copy()
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
        self.window_size = 95
        self.normalized_momentum = None
        self.columns = train_data.columns
        # self.pair_indices = [(list(self.columns).index('A'), list(self.columns).index('C')),
        #                      (list(self.columns).index('E'), list(self.columns).index('F'))]  # Indices of pairs
        self.pair_indices = []
        self.pair_trading_weight = 0.5
        self.correlation_update_frequency = 150
        self.update_momentum()  # Initialize momentum calculation
    def identify_pairs_using_correlation(self, prices):
        # Calculate daily returns
        returns = pd.DataFrame(prices[-self.correlation_update_frequency:]).pct_change().dropna()
        correlation_matrix = returns.corr()
        pairs = []
        print("Pairs:")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > 0.6: 
                    pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
                    print((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        return pairs
    def update_momentum(self):
        data_length = self.running_price_paths.shape[0]
        if data_length > 0:  # Ensure there's at least some data
            # Use available data if less than window size
            momentum_window = min(data_length, self.window_size)
            momentum = pd.DataFrame(self.running_price_paths[-momentum_window:]).pct_change().dropna().sum(axis=0)
            normalized_momentum = (momentum - momentum.mean()) / momentum.std(ddof=0)
            self.normalized_momentum = normalized_momentum.fillna(0).values  # Convert to numpy array, handle potential NaNs

    
    def allocate_portfolio(self, asset_prices):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''
        self.running_price_paths = np.vstack([self.running_price_paths, asset_prices])

        ### TODO Implement your code here
        self.update_momentum()
        # return np.array([1/6,1/6,1/6,1/6,1/6,1/6]) # Compare to naive 1/n method
        # return np.array([0.3121, -0.1765, 0.0313, 0.0574, 0.1671, 0.2556]) # Compare to some good static method
        # Calculate returns
        returns = np.diff(self.running_price_paths[-self.window_size:], axis=0) / self.running_price_paths[-self.window_size:-1]
        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns.T)

        # Objective function for MVO: minimize negative Sharpe ratio
        def objective_function(weights, mean_returns, cov_matrix):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility

        # Constraints and bounds for MVO
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'ineq', 'fun': lambda x: x + 1},
                       {'type': 'ineq', 'fun': lambda x: 1 - x}]  # Sum of weights to 1 yields the best results
        bounds = [(-1, 1) for _ in range(len(asset_prices))]  # Weights are bounded between -1 and 1

        # Initial guess for the optimizer
        initial_guess = self.normalized_momentum / np.sum(np.abs(self.normalized_momentum))

        # Perform optimization to get the optimal weights
        result = scipy.optimize.minimize(
            objective_function, initial_guess,
            args=(mean_returns, cov_matrix),
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        momentum_weights = result.x
        if (len(self.running_price_paths) % self.correlation_update_frequency == 0):
            self.pair_indices = []
            self.pair_indices = self.identify_pairs_using_correlation(self.running_price_paths)
        mean_reversion_weights = initial_guess
        for i, j in self.pair_indices:
            spread = self.running_price_paths[-self.window_size:, i] - self.running_price_paths[-self.window_size:, j]
            mean_spread = np.mean(spread)
            spread_deviation = spread - mean_spread
            adjustment = np.sign(spread_deviation[-1])
            mean_reversion_weights[i] -= adjustment
            mean_reversion_weights[j] += adjustment

        # Normalize mean reversion weights
        if np.sum(np.abs(mean_reversion_weights)) != 0:
            mean_reversion_weights /= np.sum(np.abs(mean_reversion_weights))

        # Combine the strategies
        combined_weights = momentum_weights * (1 - self.pair_trading_weight) + mean_reversion_weights * self.pair_trading_weight
        weights = combined_weights
        if not self.pair_indices:
            weights = momentum_weights
        # The optimal weights
        # weights = result.x
        weights = np.clip(weights, -1, 1)

        return weights