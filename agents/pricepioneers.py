import random
import pickle
import os
import pandas as pd
import numpy as np


'''
This template serves as a starting point for your agent.
'''

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        
        self.project_part = params['project_part'] 

        ### starting remaining inventory and inventory replenish rate are provided
        ## every time the inventory is replenished, it is set to the inventory limit
        ## the inventory_replenish rate is how often the inventory is replenished
        ## for example, we will run with inventory_replenish = 20, with the limit of 11. Then, the inventory will be replenished every 20 time steps (time steps 0, 20, 40, ...) and the inventory will be set to 11 at those time steps. 
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']

        ### useful if you want to use a more complex price prediction model
        ### note that you will need to change the name of the path and this agent file when submitting
        ### complications: pickle works with any machine learning models defined in sklearn, xgboost, etc.
        ### however, this does not work with custom defined classes, due to the way pickle serializes objects
        ### refer to './yourteamname/create_model.ipynb' for a quick tutorial on how to use pickle
        # self.filename = './[yourteamname]/trained_model'
        # self.trained_model = pickle.load(open(self.filename, 'rb'))

        ### potentially useful for Part 2 -- When competition is between two agents
        ### and you want to keep track of the opponent's status
        # self.opponent_number = 1 - agent_number  # index for opponent

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        '''
        This function updates your internal state based on the last sale that occurred.
        This template shows you several ways you can keep track of important metrics.
        '''
        ### keep track of who, if anyone, the customer bought from
        did_customer_buy_from_me = (last_sale[0] == self.this_agent_number)
        ### potentially useful for Part 2
        # did_customer_buy_from_opponent = (last_sale[0] == self.opponent_number)

        ### keep track of the prices that were offered in the last sale
        my_last_prices = last_sale[1][self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_last_prices = last_sale[1][self.opponent_number]

        ### keep track of the profit for this agent after the last sale
        my_current_profit = state[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_current_profit = state[self.opponent_number]

        ### keep track of the inventory levels after the last sale
        self.remaining_inventory = inventories[self.this_agent_number]
        ### potentially useful for Part 2
        # opponent_inventory = inventories[self.opponent_number]

        ### keep track of the time until the next replenishment
        time_until_replenish = time_until_replenish

        ### TODO - add your code here to potentially update your pricing strategy 
        ### based on what happened in the last round
        pass
    




    def action(self, obs):
        '''
        This function is called every time the agent needs to choose an action by the environment.

        The input 'obs' is a 5 tuple, containing the following information:
        -- new_buyer_covariates: a vector of length 3, containing the covariates of the new buyer.
        -- last_sale: a tuple of length 2. The first element is the index of the agent that made the last sale, if it is NaN, then the customer did not make a purchase. 
            The second element is a numpy array of length n_agents, containing the prices that were offered by each agent in the last sale.
        -- state: a vector of length n_agents, containing the current profit of each agent.
        -- inventories: a vector of length n_agents, containing the current inventory level of each agent.
        -- time_until_replenish: an integer indicating the time until the next replenishment, by which time your (and your opponent's, in part 2) remaining inventory will be reset to the inventory limit.

        The expected output is a single number, indicating the price that you would post for the new buyer.
        '''

        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        ### currently output is just a deterministic price for the item
        ### but you are expected to use the new_buyer_covariates
        ### combined with models you come up with using the training data 
        ### and history of prices from each team to set a better price for the item
        #return 30.123 #112.358
        
        # Getting the inventory level of our agent
        inventories = inventories[0]
        
        train_pricing_decisions = pd.read_csv('agents/pricepioneers/train_prices_decisions_2024.csv')
        
        # #Creating prices_to_predict array based on min and max prices in train_pricing_decisions
        # # print('min: ', np.round(train_pricing_decisions.price_item.min(), 2), 
        #     # 'max: ', np.round(train_pricing_decisions.price_item.max(), 2))
        min_price = train_pricing_decisions.price_item.min()
        max_price = train_pricing_decisions.price_item.max()

        # Generate more points near the center
        n_center = 150  # Number of central points
        center_points = np.linspace(min_price + (max_price - min_price) * 0.1,  # Start 10% above min
                                    max_price - (max_price - min_price) * 0.1,  # End 10% below max
                                    n_center)

        # Generate fewer points near the edges
        n_edges = 20  # Number of edge points
        edge_points = np.concatenate([
            np.linspace(min_price, min_price + (max_price - min_price) * 0.1, n_edges // 2),
            np.linspace(max_price - (max_price - min_price) * 0.1, max_price, n_edges // 2)
        ])

        # Combine and sort the points
        prices_to_predict = np.sort(np.concatenate([center_points, edge_points]))


        def get_prediction_logistic(fitted_model, price, covariates):
            input_data = pd.DataFrame({
                'price_item': [price],     
                'Covariate1': [covariates[0]], 
                'Covariate2': [covariates[1]],
                'Covariate3': [covariates[2]]
            })
            prediction = fitted_model.predict_proba(input_data)[:, 1]  
            return prediction[0]

        with open('agents/pricepioneers/demand_logistic_reg_kmeans_0.pkl', 'rb') as f:
            loaded_model_0 = pickle.load(f)
        with open('agents/pricepioneers/demand_logistic_reg_kmeans_1.pkl', 'rb') as f:
            loaded_model_1 = pickle.load(f)
        with open('agents/pricepioneers/demand_logistic_reg_kmeans_2.pkl', 'rb') as f:
            loaded_model_2 = pickle.load(f)
        with open('agents/pricepioneers/demand_logistic_reg_kmeans_3.pkl', 'rb') as f:
            loaded_model_3 = pickle.load(f)
        models = [loaded_model_0, loaded_model_1, loaded_model_2, loaded_model_3]

        with open('agents/pricepioneers/kmeans.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)

        # the function takes in dataframe of covariates(df), list of prices(prices_to_predict), and list of segmented logistic regression models(cluster_log_models)
        #  --> outputs demand predictions
        def get_demand_predictions_clusters(arr, prices_to_predict, cluster_log_models, kmeans_model=kmeans_model):
            # Getting predictions for train_pricing_decision to get average price
            demand_predictions = []
            covariates = [arr]
            assigned_cluster = kmeans_model.predict(covariates)[0]

            for price in prices_to_predict:
                demand_predictions.append(get_prediction_logistic(cluster_log_models[assigned_cluster], price, arr))
            
            return demand_predictions
        def dynamic_program(prices_to_predict, demand_prediction, T, K):
            demand_pred = np.array(demand_prediction)
            ratio = K/T
            if ratio>=0.9:
                ratio = 0.9
            diff = np.abs(demand_pred - ratio)
            return prices_to_predict[np.argmin(diff)]
        '''
        def get_single_step_revenue_maximizing_price_and_revenue_k(Vtplus1k, Vtplus1kminus1, price_options, demand_predictions):
            rev_list = (np.array(price_options)+Vtplus1kminus1*np.ones(len(price_options)))*np.array(demand_predictions)+(np.ones(len(demand_predictions))-demand_predictions)*Vtplus1k
            opt_index = np.argmax(rev_list)
            Ptk = price_options[opt_index]
            vtk = rev_list[opt_index]
            return Ptk, vtk
            
        def get_prices_over_time_and_expected_revenue_k(prices, demand_predictions, T, K):
            opt_price_list=np.zeros([T,K+1])
            V = np.zeros([T+1,K+1])
            for t in range(T - 1, -1, -1):
                for k in range(1, K + 1):  # We cannot sell if k = 0
                    # Optimize the price given the future value function
                    V_t_k = V[t + 1][k]
                    V_t_k_minus_1 = V[t + 1][k - 1] if k > 0 else None
                    opt_price, max_value = get_single_step_revenue_maximizing_price_and_revenue_k(V_t_k, V_t_k_minus_1, prices, demand_predictions)
                    V[t][k] = max_value  # Update the value function
                    opt_price_list[t][k] = opt_price  # Store the optimal price for time t and k items left   
            return opt_price_list, V
        '''
        threshold_1percentile = pd.read_csv('agents/pricepioneers/threshold_1percentile.csv')

        # threshold_avg = pd.read_csv('agents/pricepioneers/threshold_avg.csv')

        # threshold_10percentile0 = pd.read_csv('agents/pricepioneers/threshold_10percentile0.csv')
        # threshold_replacement0 = pd.read_csv('agents/pricepioneers/threshold_replacement0.csv')

        # threshold_10percentile1 = pd.read_csv('agents/pricepioneers/threshold_10percentile1.csv')
        # threshold_replacement1 = pd.read_csv('agents/pricepioneers/threshold_replacement1.csv')

        # threshold_10percentile2 = pd.read_csv('agents/pricepioneers/threshold_10percentile2.csv')
        # threshold_replacement2 = pd.read_csv('agents/pricepioneers/threshold_replacement2.csv')

        # threshold_10percentile3 = pd.read_csv('agents/pricepioneers/threshold_10percentile3.csv')
        # threshold_replacement3 = pd.read_csv('agents/pricepioneers/threshold_replacement3.csv')

        # thresholds = [threshold_10percentile0, threshold_10percentile1, threshold_10percentile2, threshold_10percentile3]
        # replacements = [threshold_replacement0, threshold_replacement1, threshold_replacement2, threshold_replacement3]

        def threshold_func(opt_price, t, k, threshold):
            if opt_price < threshold.iloc[k-1, 20-t]:
                # return replacement.iloc[k-1, 20-t]
                return max_price
            else:
                return opt_price

        demand_prediction = get_demand_predictions_clusters(new_buyer_covariates, prices_to_predict, models, kmeans_model=kmeans_model)
        # for price in prices_to_predict:
        #     demand_prediction.append(get_prediction_logistic(loaded_model, price, new_buyer_covariates))
        opt_price =  dynamic_program(prices_to_predict, demand_prediction, T=21-time_until_replenish, K=inventories)
        # opt_price = threshold_func(opt_price, time_until_replenish, inventories, threshold_1percentile)
        return opt_price

