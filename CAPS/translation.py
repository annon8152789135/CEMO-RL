import numpy as np
import math
from CAPS.qm import QuineMcCluskey
from CAPS.condense_ex import Explainer

class PredicateTemplate:
    def __init__(self, num_feats):
        self.num_feats = num_feats
        self.attr_names = None
    
    def predicate_set(self):
        raise NotImplementedError

    def feat_groups(self):
        raise NotImplementedError

    def translate_state(self, state):
        raise NotImplementedError
    
    def num_predicates(self):
        raise NotImplementedError
    
    def state_to_binary(self, state):
        raise NotImplementedError
    
    def reduce_logic(self, c_binary):
        #Sections of code adapted from https://gitlab.tue.nl/ha800-hri/hayes-shah/-/blob/master/hayes_shah/hs.py
        """
        c_binary: The set of abstract classes. An array of length num_abstract_classes
            c_binary[k] is an array which contains all states of the kth class
            that array contains tuples of the form (binary_state, action)
        Simplify the predicates of an abstract state into the minimal form
        This algorithm from Hayes and Shah 2017 requires the predicate features from all binary states in the target
        class (the abstract state, in this case), and the non-target class (all other abstract states)
        The target class and non-target classes should be mutually exclusive for the algorithm
        This is not reasonable to assume, given that binary states that are the same could be in different classes
            due to stochasticity in the policy and inaccuracies in the predicates
        So, the initial solution would be to loop through the other abstract states and remove the predicate states
            which are the exact same. The resulting explanations may not be completely faithful, but higher accuracy would
            come with better predicates/
        There could be an issue where no reliable predicate explanation is found, since there are just so many states in the
            other classes. In that case, I would come up with my own way of simplifying the predicates
        """

        qm = QuineMcCluskey()

        use_qm = True

        predicates = self.predicate_set()
        explanations = []
        condensed_sets = []

        for i, abs_class in enumerate(c_binary):
            target_states = [t[0] for t in abs_class]
            
            non_target_states = []
            for j in range(len(c_binary)):
                if j != i:
                    for k in range(len(c_binary[j])):
                        s = c_binary[j][k][0]
                        non_target_states.append(s)
            

            for s in target_states: #Loop through sets to ensure the intersection is empty
                for j, non_s in enumerate(non_target_states):
                    if np.array_equal(s, non_s):
                        non_target_states.pop(j)
            
            
            if use_qm:
                target_state_strings = []
                non_target_state_strings = []
        
                for s in target_states:
                    string = ''
                    for f in s:
                        string = string + str(f)
                    target_state_strings.append(string)
        
                
                for s in non_target_states:
                    string = ''
                    for f in s:
                        string = string + str(f)
                    non_target_state_strings.append(string)
                    
                
                n = len(target_states[0])
                all_bin = [bin(x)[2:].rjust(n, '0') for x in range(2**n)]
                not_valid = list(set(all_bin) - set(target_state_strings) - set(non_target_state_strings)) #All states which never appear

                
                
                minterms = qm.simplify_los(target_state_strings, not_valid)
                print('{}: {}'.format(i+1,minterms))
                clauses = self.minterm_to_clause(minterms, predicates)
                print('{}: {}'.format(i+1,clauses))
                explanations.append(' or '.join(clauses))
            
            else:
                proportions = np.zeros(self.num_predicates())
                for s in target_states:
                    proportions = proportions + np.array(s)
                pos_proportions = proportions / len(target_states)
                pos_explans = []
                neg_explans = []
                for j, p in enumerate(pos_proportions):
                    predicate = predicates[j]
                    if p >= 0.9:
                        pos_explans.append(predicate['true'])
                    elif p <= 0.1:
                        neg_explans.append(predicate['false'])
                
                if pos_explans == []:
                    most_common = np.argmax(pos_proportions) #Add most common occurence
                    neg_explans.append(predicates[most_common]['true'])
                    explanations.append(' and '.join(neg_explans)) #Only use neg explans if no pos exist
                else:
                    explanations.append(' and '.join(pos_explans))
                condensed_sets.append(pos_proportions)

        return explanations

    def minterm_to_clause(self, minterms, predicates):
        

        clauses = []

        for min_term in minterms:
            str_terms = []
            for i in range(len(min_term)):
                predicate = predicates[i]
                if min_term[i] == '0':
                    str_terms.append(predicate['false'])
                elif min_term[i] == '1':
                    str_terms.append(predicate['true'])

            clauses.append(' and '.join(str_terms))

        return clauses
    

    def my_translation_algo(self, c_binary):
        predicates = self.predicate_set()
        explanations = []
        condensed_sets = []

    
        for i, abs_class in enumerate(c_binary):
            target_states = [t[0] for t in abs_class]
            if len(target_states) == 0:
                explanations.append('(no states)')
                continue
            e = Explainer(target_states, self.feat_groups(), len(predicates), predicates)
            explanations.append(e.full_translate())
        
        return explanations

class LunarLanderPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['X Coordinate',
                           'Y Coordinate',
                           'X Velocity',
                           'Y Velocity',
                           'Lander Angle',
                           'Angular Velocity',
                           'Left leg on ground',
                           'Right leg on ground']
        self.language_set = np.array(['Left of the goal', 'Right of the goal',
                                      'On top of goal', 'Higher than goal', 'Same height as goal',
                                      'Left leg on ground', 'Right leg on ground',
                                      'Lander tilted left', 'Lander tilted right',
                                      'Moving right', 'Moving left'])
    
    def state_to_binary(self, state):
        b = [self.left_of_goal(state),
             self.right_of_goal(state),
             self.on_top_of_goal(state),
             self.higher_than_goal(state),
             self.at_same_height(state),
             self.left_leg_on_ground(state),
             self.right_leg_on_ground(state),
             self.tilted_left(state),
             self.tilted_right(state),
             self.moving_right(state),
             self.moving_left(state)]
        return np.array(b)
    
    def translate_state(self, binary_set):
        idx = np.where(binary_set==1)[0]
        true_set = self.language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string

    def feat_groups(self):
        groups = [[0, 1, 2], [3, 4], [5], [6], [7, 8], [9, 10]]
        return groups
    
    def predicate_set(self):
        predicates = [{'true': 'Left of the goal', 'false': 'Not left of the goal'},
                      {'true': 'Right of the goal', 'false': 'Not right of the goal'},
                      {'true': 'Directly on top of goal', 'false': 'Not directly on top of goal'},
                      {'true': 'Higher than goal', 'false': 'Not higher than goal'},
                      {'true': 'At same height as goal', 'false': 'Not at same height as goal'},
                      {'true': 'Left leg on the ground', 'false': 'Left leg not on the ground'},
                      {'true': 'Right leg on the ground', 'false': 'Right leg not on the ground'},
                      {'true': 'Lander tilted left', 'false': 'Lander not tilted left'},
                      {'true': 'Lander tilted right', 'false': 'Lander not tilted right'},
                      {'true': 'Moving right', 'false': 'Not moving right'},
                      {'true': 'Moving left', 'false': 'Not moving left'}]
        return predicates
    
    def num_predicates(self):
        return len(self.predicate_set())
    
    def left_of_goal(self, state):
        if state[0] < -0.08:
            return 1
        else:
            return 0
    
    def right_of_goal(self, state):
        if state[0] > 0.08:
            return 1
        else:
            return 0
    
    def on_top_of_goal(self, state):
        if np.abs(state[0]) <= 0.08:
            return 1
        else:
            return 0
    
    def higher_than_goal(self, state):
        if state[1] > 0.08:
            return 1
        else:
            return 0
    
    def at_same_height(self, state):
        if state[1] <= 0.08:
            return 1
        else:
            return 0
    
    def left_leg_on_ground(self, state):
        if state[6] == 1:
            return 1
        else:
            return 0
    
    def right_leg_on_ground(self, state):
        if state[7] == 1:
            return 1
        else:
            return 0
    
    def tilted_left(self, state):
        if state[4] < -0.3:
            return 1
        else:
            return 0
    def tilted_right(self, state):
        if state[4] > 0.3:
            return 1
        else:
            return 0
    
    def moving_right(self, state):
        if state[2] > 0.01:
            return 1
        else:
            return 0
    
    def moving_left(self, state):
        if state[2] < -0.01:
            return 1
        else:
            return 0

class BlackjackPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Current sum', 'Dealer card', 'Usable ace']
        self.language_set = np.array(['sum less than 14', 'sum 14-16','sum 17-19',
                                      'sum 20-21', ' d sum less 7', 'd sum 7-9',
                                      'd sum 10-ace','ace 11'])
    

    def predicate_set(self):
        predicates = [{'true': 'Sum less than 14', 'false': 'Sum not less than 14'},
                      {'true': 'Sum of 14-16', 'false': 'Sum not of 14-16'},
                      {'true': 'Sum of 17-19', 'false': 'Sum not of 17-19'},
                      {'true': 'Sum of 20-21', 'false': 'Sum not of 20-21'},
                      {'true': 'Dealer card less than 7', 'false': 'Dealer card 7 or more'},
                      {'true': 'Dealer card 7-9', 'false': 'Dealer card not 7-9'},
                      {'true': 'Dealer card 10 or ace', 'false': 'Dealer card not 10 or ace'},
                      {'true': 'Ace is 11', 'false': 'No ace or ace is not 11'}]
        
        return predicates
    
    def num_predicates(self):
        return len(self.predicate_set())
    
    def feat_groups(self):
        groups = [[0, 1, 2, 3], [4, 5, 6], [7]]
        return groups


    def state_to_binary(self, state):
        b = [self.less_14(state),
             self.p14_16(state),
             self.p17_19(state),
             self.p20_21(state),
             self.dless_7(state),
             self.d7_9(state),
             self.d10_ace(state),
             self.use_ace(state)]
        return np.array(b)
    
    def translate_state(self, binary_set):
        idx = np.where(binary_set==1)[0]
        true_set = self.language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string


    def less_14(self, state):
        if state[0] < 14:
            return 1
        else:
            return 0
    
    def p14_16(self, state):
        if state[0] >= 14 and state[0] <= 16:
            return 1
        else:
            return 0
    
    def p17_19(self, state):
        if state[0] >= 17 and state[0] <= 19:
            return 1
        else:
            return 0
    
    def p20_21(self, state):
        if state[0] == 20 or state[0] == 21:
            return 1
        else:
            return 0

    def dless_7(self, state):
        if state[1] < 7 and state[1] != 1:
            return 1
        else:
            return 0
    
    def d7_9(self, state):
        if state[1] >= 7 and state[1] <= 9:
            return 1
        else:
            return 0
    
    def d10_ace(self, state):
        if state[1] == 10 or state[1] == 1:
            return 1
        else:
            return 0
    
    
    def use_ace(self, state):
        if state[2] == 1:
            return 1
        else:
            return 0
    
class MountainCarPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Car Position', 'Car Velocity']
        self.language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slowly',
                    'Moving right slowly',
                    'Not moving',
                    'Moving left quickly',
                    'Moving right quickly'])

    def predicate_set(self):
        predicates = [{'true': 'At the bottom', 'false': 'Not at the bottom'},
                      {'true': 'On the left slope', 'false': 'Not on the left slope'},
                      {'true': 'On the right slope', 'false': 'Not on the right slope'},
                      {'true': 'High up on the left slope', 'false': 'Not high up on the left slope'},
                      {'true': 'High up on the right slope', 'false': 'Not high up on the right slope'},
                      {'true': 'Moving left slowly', 'false': 'Not moving left slowly'},
                      {'true': 'Moving right slowly', 'false': 'Not moving right slowly'},
                      {'true': 'Not moving', 'false': 'Not moving'},
                      {'true': 'Moving left quickly', 'false': 'Not moving left quickly'},
                      {'true': 'Moving right quickly', 'false': 'Not moving right quickly'}]
        return predicates

    def state_to_binary(self, state):
        binary_set = [self.at_bottom(state),
                      self.on_left_slope(state),
                      self.on_right_slope(state),
                      self.high_on_left(state),
                      self.high_on_right(state),
                      self.moving_left_slow(state),
                      self.moving_right_slow(state),
                      self.not_moving(state),
                      self.moving_left_fast(state),
                      self.moving_right_fast(state)]
        
        return np.array(binary_set)
    
    def translate_state(self, binary_set):
        language_set = np.array(['At the bottom',
                    'On the left slope',
                    'On the right slope',
                    'High up on the left slope',
                    'High up on the right slope',
                    'Moving left slow',
                    'Moving right slow',
                    'Not moving',
                    'Moving left fast',
                    'Moving right fast'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string

    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        return groups

    def num_predicates(self):
        return 10
    
    def at_bottom(self, state):
        if state[0] >= -0.6 and state[0] <= -0.4:
            return 1
        else:
            return 0
    
    def on_left_slope(self, state):
        if state[0] < -0.6 and state[0] > -0.9:
            return 1
        else:
            return 0
    
    def on_right_slope(self, state):
        if state[0] > -0.4 and state[0] < 0.3:
            return 1
        else:
            return 0
    
    def high_on_left(self, state):
        if state[0] <= -0.9:
            return 1
        else:
            return 0
    
    def high_on_right(self, state):
        if state[0] >= 0.3:
            return 1
        else:
            return 0
    
    def moving_left_slow(self, state):
        if state[1] < 0 and state[1] > -0.025:
            return 1
        else:
            return 0
    
    def moving_right_slow(self, state):
        if state[1] > 0 and state[1] < 0.025:
            return 1
        else:
            return 0
    
    def not_moving(self, state):
        if state[1] == 0:
            return 1
        else:
            return 0
    
    def moving_left_fast(self, state):
        if state[1] <= -0.025:
            return 1
        else:
            return 0
    
    def moving_right_fast(self, state):
        if state[1] >= 0.025:
            return 1
        else:
            return 0

class GridworldPredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Position']
        self.language_set = np.array(['At the start',
                    'Reached the goal',
                    'A cliff is below',
                    'On the left border',
                    'On the right border',
                    'In free space',
                    'Near the goal'])

    def state_to_coords(self, state):
        obs = state[0]
        coords = (obs // 12, obs % 12)
        return coords

    def predicate_set(self):
        predicates = [{'true': 'At the start', 'false': 'Not at the start'},
                      {'true': 'Reached the goal', 'false': 'Has not reached the goal'},
                      {'true': 'A cliff is below', 'false': 'A cliff is not below'},
                      {'true': 'On the left border', 'false': 'Not on the left border'},
                      {'true': 'On the right border', 'false': 'Not on the right border'},
                      {'true': 'In free space', 'false': 'Not in free space'},
                      {'true': 'Near the goal', 'false': 'Not near the goal'}]
    
        return predicates
    
    def state_to_binary(self, state):
        coords = self.state_to_coords(state)
        binary_set = [self.at_start(coords),
                      self.at_goal(coords),
                      self.cliff_below(coords),
                      self.at_left_edge(coords),
                      self.at_right_edge(coords),
                      self.in_free_space(coords),
                      self.near_goal(coords)]
        
        return np.array(binary_set)
    
    def translate_state(self, binary_set):
        language_set = np.array(['At the start',
                    'Reached the goal',
                    'A cliff is below',
                    'On the left border',
                    'On the right border',
                    'In free space',
                    'Near the goal'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string


    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4, 5], [6]]
        return groups

    def num_predicates(self):
        return 7

    def at_start(self, coords):
        if coords[0] == 3 and coords[1] == 0:
            return 1
        else:
            return 0
    
    def at_goal(self, coords):
        if coords[0] == 3 and coords[1] == 11:
            return 1
        else:
            return 0

    def cliff_below(self, coords):
        if coords[0] == 2 and coords[1] > 0 and coords[1] < 11:
            return 1
        else:
            return 0
    
    def at_left_edge(self, coords):
        if coords[0] < 3 and coords[1] == 0:
            return 1
        else:
            return 0
    
    def at_right_edge(self, coords):
        if coords[0] < 3 and coords[1] == 11:
            return 1
        else:
            return 0
    
    def in_free_space(self, coords):
        if coords[0] < 2 and coords[1] > 0 and coords[1] < 11:
            return 1
        else:
            return 0

    def near_goal(self, coords):
        if coords[0] > 1 and coords[1] > 9:
            return 1
        else:
            return 0

class CartpolePredicates(PredicateTemplate):
    def __init__(self, num_feats):
        super().__init__(num_feats)
        self.attr_names = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
        self.language_set = np.array(['Pole is falling to the left',
                    'Pole is falling to the right',
                    'Pole is stabilizing from left',
                    'Pole is stabilizing from right',
                    'Pole is standing up',
                    'Cart is moving left',
                    'Cart is moving right',
                    'Cart is on the left',
                    'Cart is on the right',
                    'Cart is in the middle'])

    def predicate_set(self):
        predicates = [{'true': 'Pole is falling to the left', 'false': 'Pole is not falling to the left'},
                      {'true': 'Pole is falling to the right', 'false': 'Pole is not falling to the right'},
                      {'true': 'Pole is stabilizing to the left', 'false': 'Pole is not stabilizing to the left'},
                      {'true': 'Pole is stabilizing to the right', 'false': 'Pole is not stabilizing to the right'},
                      {'true': 'Pole is standing up', 'false': 'Pole is not standing up'},
                      {'true': 'Cart is moving left', 'false': 'Cart is not moving left'},
                      {'true': 'Cart is moving right', 'false': 'Cart is not moving right'},
                      {'true': 'Cart is on the left', 'false': 'Cart is not on the left'},
                      {'true': 'Cart is on the right', 'false': 'Cart is not on the right'},
                      {'true': 'Cart is in the middle', 'false': 'Cart is not in the middle'}]
    
        return predicates
    
    def translate_state(self, binary_set):
        language_set = np.array(['Pole is falling to the left',
                    'Pole is falling to the right',
                    'Pole is stabilizing from left',
                    'Pole is stabilizing from right',
                    'Pole is standing up',
                    'Cart is moving left',
                    'Cart is moving right',
                    'Cart is on the left',
                    'Cart is on the right',
                    'Cart is in the middle'])
        
        idx = np.where(binary_set==1)[0]
        true_set = language_set[idx]
        string = ''
        if true_set.size != 0:
            string = true_set[0]
            if true_set[1:].size != 0:
                for pred in true_set[1:]:
                    string = string + ' and '
                    string = string + pred
        
        return string
    
    def feat_groups(self):
        groups = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
        return groups

    def num_predicates(self):
        return 10
    
    def state_to_binary(self, state):
        state = np.reshape(state, [-1])
        binary = [self.pole_fall_left(state),
                  self.pole_fall_right(state),
                  self.pole_stabilize_left(state),
                  self.pole_stabilize_right(state),
                  self.pole_standing_up(state),
                  self.cart_moving_left(state),
                  self.cart_moving_right(state),
                  self.cart_pos_left(state),
                  self.cart_pos_right(state),
                  self.cart_near_middle(state)]
        
        return np.array(binary)
    
    def pole_fall_left(self, state):
        if state[2] < -0.01 and state[3] < 0:
            return 1
        else:
            return 0
    
    def pole_fall_right(self, state):
        if state[2] > 0.01 and state[3] > 0:
            return 1
        else:
            return 0
    
    def pole_stabilize_left(self, state):
        if state[2] < -0.01 and state[3] > 0:
            return 1
        else:
            return 0
    
    def pole_stabilize_right(self, state):
        if state[2] > 0.01 and state[3] < 0:
            return 1
        else:
            return 0
    
    def pole_standing_up(self, state):
        if np.abs(state[2]) <= 0.01:
            return 1
        else:
            return 0
    
    def cart_moving_left(self, state):
        if state[1] < 0:
            return 1
        else:
            return 0
    
    def cart_moving_right(self, state):
        if state[1] >= 0:
            return 1
        else:
            return 0
    
    def cart_pos_left(self, state):
        if state[0] < -0.05:
            return 1
        else:
            return 0
    
    def cart_pos_right(self, state):
        if state[0] > 0.05:
            return 1
        else:
            return 0
    
    def cart_near_middle(self, state):
        if np.abs(state[0]) < 0.05:
            return 1
        else:
            return 0

class AcornPredicates(PredicateTemplate):
    def __init__(self, num_feats, num_bins=10):
        super().__init__(num_feats)
        self.num_bins = num_bins
        self.predicate_set = []
        self.binned_indices = []

        self.agent1_features = ['Percent Tweets', 'Percent Replies',
                                'Percent Retweets', 'Percent Mentions',
                                'Percent Followers']

    
    def translate_state(self, state): #Temporary implementation. Later, will include other ways of predicate grounding
        self.predicate_set = []
        binned = self.binning(state)
        for i, feature in enumerate(binned):
            self.binned_indices.append(len(self.predicate_set))
            self.predicate_set.append(feature)
        
        nl_predicate_set = self.nl_grounding()

        return self.predicate_set, nl_predicate_set

    def state_to_binary(self, state):
        return self.binning(state)
    
    def binning(self, state):
        binned_state = []
        for feature in state:
            assert 0 <= feature and feature <= 1

            idx = math.floor(feature * self.num_bins)
            binary = np.zeros(self.num_bins)
            binary[idx] = 1
            binned_state.append(binary)
        binned_state = np.reshape(np.array(binned_state), [-1])
    
        return binned_state
    
    def nl_grounding(self): #Temporary. Later, will include translations for other types of predicates
        for i, pred in enumerate(self.predicate_set):
            if i in self.binned_indices:
                feat_idx = np.where(np.array(self.binned_indices==i))[0][0]
                self.nl_predicate_set.append(self.translate_bins(pred, feat_idx))
        
        return self.nl_predicate_set
        
        

    
    def translate_bins(self, predicate, feat_idx):
        idx = np.argmax(predicate)
        low_bound = idx / self.num_bins
        high_bound = low_bound + (1 / self.num_bins)

        string = "{} is between {} and {}".format(self.agent1_features[feat_idx], low_bound, high_bound)
        return string

    def num_predicates(self):
        return self.num_feats * self.num_bins

class FruitTreePredicates(PredicateTemplate):
    def __init__(self, num_feats, depth=5):
        super().__init__(num_feats)
        self.depth = depth
        self.attr_names = ['Level', 'Position']
        self.language_set = np.array([
            # depth / special
            'At root', 'At leaf', 'Near leaf',

            # vertical bands (6)
            'In vertical band 1', 'In vertical band 2', 'In vertical band 3',
            'In vertical band 4', 'In vertical band 5', 'In vertical band 6',

            # edges
            'On left edge', 'On right edge',

            # horizontal bands (6)
            'On horizontal band 1', 'On horizontal band 2', 'On horizontal band 3',
            'On horizontal band 4', 'On horizontal band 5', 'On horizontal band 6',

            # parity
            'At even level', 'At odd level'
        ])

    def state_to_binary(self, state):
        lvl, pos = int(state[0]), int(state[1])
        b = [
            self.at_root((lvl, pos)),                 # 0
            self.at_leaf((lvl, pos)),                 # 1
            self.near_leaf((lvl, pos)),               # 2

            self.in_vertical_band_1((lvl, pos)),      # 3
            self.in_vertical_band_2((lvl, pos)),      # 4
            self.in_vertical_band_3((lvl, pos)),      # 5
            self.in_vertical_band_4((lvl, pos)),      # 6
            self.in_vertical_band_5((lvl, pos)),      # 7
            self.in_vertical_band_6((lvl, pos)),      # 8

            self.on_left_edge((lvl, pos)),            # 9
            self.on_right_edge((lvl, pos)),           # 10

            self.on_horizontal_band_1((lvl, pos)),    # 11
            self.on_horizontal_band_2((lvl, pos)),    # 12
            self.on_horizontal_band_3((lvl, pos)),    # 13
            self.on_horizontal_band_4((lvl, pos)),    # 14
            self.on_horizontal_band_5((lvl, pos)),    # 15
            self.on_horizontal_band_6((lvl, pos)),    # 16

            self.even_level((lvl, pos)),              # 17
            self.odd_level((lvl, pos)),               # 18
        ]

        if b[17] + b[18] != 1:
            raise ValueError(
                f"Parity error at state (lvl={lvl}, pos={pos}): even={b[17]}, odd={b[18]}"
            )
        return np.array(b, dtype=int)

    def translate_state(self, binary_set):
        idx = np.where(binary_set == 1)[0]
        true_set = self.language_set[idx]
        if true_set.size == 0:
            return ''
        return ' and '.join(true_set.tolist())

    def feat_groups(self):
        groups = [
            [0, 1, 2],                    # depth: root/leaf/near-leaf
            [3, 4, 5, 6, 7, 8],           # vertical bands: 6
            [9, 10],                      # edges
            [11, 12, 13, 14, 15, 16],     # horizontal bands: 6
            [17, 18],                     # parity of level
        ]
        return groups

    def predicate_set(self):
        predicates = [
            {'true': 'At root', 'false': 'Not at root'},
            {'true': 'At leaf', 'false': 'Not at leaf'},
            {'true': 'Near leaf', 'false': 'Not near leaf'},

            {'true': 'In vertical band 1', 'false': 'Not in vertical band 1'},
            {'true': 'In vertical band 2', 'false': 'Not in vertical band 2'},
            {'true': 'In vertical band 3', 'false': 'Not in vertical band 3'},
            {'true': 'In vertical band 4', 'false': 'Not in vertical band 4'},
            {'true': 'In vertical band 5', 'false': 'Not in vertical band 5'},
            {'true': 'In vertical band 6', 'false': 'Not in vertical band 6'},

            {'true': 'On left edge', 'false': 'Not on left edge'},
            {'true': 'On right edge', 'false': 'Not on right edge'},

            {'true': 'On horizontal band 1', 'false': 'Not on horizontal band 1'},
            {'true': 'On horizontal band 2', 'false': 'Not on horizontal band 2'},
            {'true': 'On horizontal band 3', 'false': 'Not on horizontal band 3'},
            {'true': 'On horizontal band 4', 'false': 'Not on horizontal band 4'},
            {'true': 'On horizontal band 5', 'false': 'Not on horizontal band 5'},
            {'true': 'On horizontal band 6', 'false': 'Not on horizontal band 6'},

            {'true': 'At even level', 'false': 'Not at even level'},
            {'true': 'At odd level', 'false': 'Not at odd level'}
        ]
        return predicates

    def num_predicates(self):
        return len(self.predicate_set())

    # ---- Helpers on the (level, position) topology ----
    def _width_at(self, lvl: int) -> int:
        # number of nodes at this level
        return 1 << lvl  # 2**lvl

    def get_vertical_sixths(self):
        """
        Split levels into 6 vertical bands as evenly as possible.
        Uses a bucket size based on floor division, similar to your quarters logic.
        """
        s = max(1, self.depth // 6)
        b1 = s
        b2 = 2 * s
        b3 = 3 * s
        b4 = 4 * s
        b5 = 5 * s
        return b1, b2, b3, b4, b5

    # Depth/level predicates
    def at_root(self, state):
        lvl, _ = state
        return int(lvl == 0)

    def at_leaf(self, state):
        lvl, _ = state
        return int(lvl == self.depth - 1)

    def near_leaf(self, state):
        lvl, _ = state
        return int(lvl >= max(1, self.depth - 2) and lvl < self.depth - 1)

    # Vertical banding (6 bands)
    def in_vertical_band_1(self, state):
        lvl, _ = state
        b1, _, _, _, _ = self.get_vertical_sixths()
        return int(lvl < b1)

    def in_vertical_band_2(self, state):
        lvl, _ = state
        b1, b2, _, _, _ = self.get_vertical_sixths()
        return int(lvl >= b1 and lvl < b2)

    def in_vertical_band_3(self, state):
        lvl, _ = state
        _, b2, b3, _, _ = self.get_vertical_sixths()
        return int(lvl >= b2 and lvl < b3)

    def in_vertical_band_4(self, state):
        lvl, _ = state
        _, _, b3, b4, _ = self.get_vertical_sixths()
        return int(lvl >= b3 and lvl < b4)

    def in_vertical_band_5(self, state):
        lvl, _ = state
        _, _, _, b4, b5 = self.get_vertical_sixths()
        return int(lvl >= b4 and lvl < b5)

    def in_vertical_band_6(self, state):
        lvl, _ = state
        _, _, _, _, b5 = self.get_vertical_sixths()
        return int(lvl >= b5)

    # Horizontal structure predicates (6 bands)
    def on_left_edge(self, state):
        lvl, pos = state
        return int(pos == 0 and lvl > 0)

    def on_right_edge(self, state):
        lvl, pos = state
        return int(lvl > 0 and pos == self._width_at(lvl) - 1)

    def _horizontal_thresholds(self, lvl):
        w = self._width_at(lvl)
        t1 = w // 6
        t2 = (2 * w) // 6
        t3 = (3 * w) // 6
        t4 = (4 * w) // 6
        t5 = (5 * w) // 6
        return w, t1, t2, t3, t4, t5

    def on_horizontal_band_1(self, state):
        lvl, pos = state
        if lvl == 0:
            return 0
        w, t1, _, _, _, _ = self._horizontal_thresholds(lvl)
        return int(pos < t1 and lvl > 0)

    def on_horizontal_band_2(self, state):
        lvl, pos = state
        if lvl == 0:
            return 0
        w, t1, t2, _, _, _ = self._horizontal_thresholds(lvl)
        return int(pos >= t1 and pos < t2 and lvl > 0)

    def on_horizontal_band_3(self, state):
        lvl, pos = state
        if lvl == 0:
            return 0
        w, _, t2, t3, _, _ = self._horizontal_thresholds(lvl)
        return int(pos >= t2 and pos < t3 and lvl > 0)

    def on_horizontal_band_4(self, state):
        lvl, pos = state
        if lvl == 0:
            return 0
        w, _, _, t3, t4, _ = self._horizontal_thresholds(lvl)
        return int(pos >= t3 and pos < t4 and lvl > 0)

    def on_horizontal_band_5(self, state):
        lvl, pos = state
        if lvl == 0:
            return 0
        w, _, _, _, t4, t5 = self._horizontal_thresholds(lvl)
        return int(pos >= t4 and pos < t5 and lvl > 0)

    def on_horizontal_band_6(self, state):
        lvl, pos = state
        if lvl == 0:
            return 0
        w, _, _, _, _, t5 = self._horizontal_thresholds(lvl)
        return int(pos >= t5 and lvl > 0)

    # Parity of level
    def even_level(self, state):
        lvl, _ = state
        return int(lvl % 2 == 0)

    def odd_level(self, state):
        lvl, _ = state
        return int(lvl % 2 == 1)



class DeepSeaPredicates(PredicateTemplate):
    """
    Predicates for Deep Sea Treasure–style grids.

    Assumptions:
      - State is (row, col) with row=0 at the surface (top) and row=H-1 at seabed (bottom).
      - Grid shape is (height, width). Defaults to (11, 11) like the classic DST.
    """
    def __init__(self, num_feats, grid_shape=(11, 11)):
        super().__init__(num_feats)
        self.H, self.W = grid_shape
        self.attr_names = ['Row', 'Col']
        self.language_set = np.array([
            'At surface', 'At seabed', 'Near seabed', 'Near surface',
            'In upper sea', 'In middle sea', 'In lower sea',
            'On left wall', 'On right wall',
            'In left third', 'In central third', 'In right third',
            'At even row', 'At odd row'
        ])

    # ---- Utilities ----
    def _third_bounds_vertical(self):
        # Split rows into thirds (upper/middle/lower) as evenly as possible
        upper = max(1, self.H // 3)
        lower = 2 * upper
        # If H not divisible by 3, lower may be < H; that’s fine (last bucket gets the remainder)
        return upper, lower

    def _third_bounds_horizontal(self):
        left = max(1, self.W // 3)
        right = 2 * left
        return left, right

    # ---- Encoding / Decoding ----
    def state_to_binary(self, state):
        row, col = int(state[0]), int(state[1])
        b = [
            self.at_surface((row, col)),
            self.near_surface((row, col)),

            self.at_seabed((row, col)),
            self.near_seabed((row, col)),

            self.in_upper_sea((row, col)),
            self.in_middle_sea((row, col)),
            self.in_lower_sea((row, col)),

            self.on_left_wall((row, col)),
            self.on_right_wall((row, col)),

            self.in_left_third((row, col)),
            self.in_central_third((row, col)),
            self.in_right_third((row, col)),

            self.even_row((row, col)),
            self.odd_row((row, col)),
        ]
        return np.array(b, dtype=int)

    def translate_state(self, binary_set):
        idx = np.where(binary_set == 1)[0]
        true_set = self.language_set[idx]
        if true_set.size == 0:
            return ''
        return ' and '.join(true_set.tolist())

    # ---- Groups (for mutually exclusive sets etc.) ----
    def feat_groups(self):
        groups = [
            [0, 2],      # vertical extremes: surface/seabed
            [1, 3],      # near-surface/near-seabed 
            [4, 5, 6],      # vertical thirds
            [7, 8],         # lateral walls
            [9, 10, 11],     # horizontal thirds
            [12, 13],       # row parity
        ]
        return groups

    def predicate_set(self):
        return [
            {'true': 'At surface', 'false': 'Not at surface'},
            {'true': 'Near surface', 'false': 'Not near surface'},

            {'true': 'At seabed', 'false': 'Not at seabed'},
            {'true': 'Near seabed', 'false': 'Not near seabed'},
            

            {'true': 'In upper sea', 'false': 'Not in upper sea'},
            {'true': 'In middle sea', 'false': 'Not in middle sea'},
            {'true': 'In lower sea', 'false': 'Not in lower sea'},

            {'true': 'On left wall', 'false': 'Not on left wall'},
            {'true': 'On right wall', 'false': 'Not on right wall'},

            {'true': 'In left third', 'false': 'Not in left third'},
            {'true': 'In central third', 'false': 'Not in central third'},
            {'true': 'In right third', 'false': 'Not in right third'},

            {'true': 'At even row', 'false': 'Not at even row'},
            {'true': 'At odd row', 'false': 'Not at odd row'},
        ]

    def num_predicates(self):
        return len(self.predicate_set())

    # ---- Predicates (vertical structure) ----
    def at_surface(self, state):
        row, _ = state
        return int(row == 0)

    def at_seabed(self, state):
        row, _ = state
        return int(row == self.H - 1)

    def near_seabed(self, state):
        row, _ = state
        # one or two rows above seabed, depending on grid height
        k = max(1, min(2, self.H // 5))  # adaptive band for small/large grids
        return int(self.H - 1 - k <= row < self.H - 1)

    def near_surface(self, state):
        row, _ = state
        # one or two rows below the surface, depending on grid height
        k = max(1, min(2, self.H // 5))  # small band near the top
        return int(0 < row <= k)

    # Vertical thirds (upper/middle/lower by row index)
    def in_upper_sea(self, state):
        row, _ = state
        up, _ = self._third_bounds_vertical()
        return int(row < up)

    def in_middle_sea(self, state):
        row, _ = state
        up, low = self._third_bounds_vertical()
        return int(up <= row < low)

    def in_lower_sea(self, state):
        row, _ = state
        _, low = self._third_bounds_vertical()
        return int(row >= low)

    # ---- Predicates (horizontal structure) ----
    def on_left_wall(self, state):
        _, col = state
        return int(col == 0)

    def on_right_wall(self, state):
        _, col = state
        return int(col == self.W - 1)

    def in_left_third(self, state):
        _, col = state
        if self.W <= 2:
            return 0
        L, _ = self._third_bounds_horizontal()
        return int(col < L)

    def in_central_third(self, state):
        _, col = state
        if self.W <= 2:
            return 0
        L, R = self._third_bounds_horizontal()
        return int(L <= col < R)

    def in_right_third(self, state):
        _, col = state
        if self.W <= 2:
            return 0
        _, R = self._third_bounds_horizontal()
        return int(col >= R)

    # ---- Parity ----
    def even_row(self, state):
        row, _ = state
        return int(row % 2 == 0)

    def odd_row(self, state):
        row, _ = state
        return int(row % 2 == 1)

class HighwayPredicates(PredicateTemplate):
    """
    Predicates for mo-highway-v0 with Kinematics observation (shape = (5, 5)).
    Each row is a vehicle: [x, y, vx, vy, heading], with row 0 the ego.
    - x: longitudinal position (m), ego-centered (ego near 0)
    - y: lateral position (lane-like units), ego lane near integer
    - vx: longitudinal speed (m/s)
    - vy: lateral speed (m/s)
    - heading: radians (0 aligned with road)
    """
    def __init__(
        self,
        num_feats,
        lanes_count=4,
        lane_width=1.0,
        # speed band based on reward_speed_range ~ [20,30] m/s by default
        slow_speed_max=10.0,
        fast_speed_min=30.0,
        # lateral/neighbor inference thresholds
        lane_tol_frac=1,              # how tight is "same lane": |Δy| < lane_width * lane_tol_frac
        close_front_gap=30.0,           # meters considered "close" ahead
        close_rear_gap=25.0,            # meters considered "close" behind
        lane_change_vy=0.02,            # |vy| above this → changing lanes
        x_span=100.0                  # consider neighbors within ±x_span meters longitudinally
    ):
        super().__init__(num_feats)
        self.lanes_count = lanes_count
        self.lane_width = lane_width
        self.slow_speed_max = slow_speed_max
        self.fast_speed_min = fast_speed_min
        self.lane_tol_frac = lane_tol_frac
        self.lane_tol = lane_width * lane_tol_frac
        self.close_front_gap = close_front_gap
        self.close_rear_gap = close_rear_gap
        self.lane_change_vy = lane_change_vy
        self.x_span = x_span

        self.attr_names = ['Vehicles(5×5)']
        self.language_set = np.array([
            # Ego speed bands
            'Ego slow', 'Ego medium', 'Ego fast',
            # Ego lane position
            'In leftmost lane', 'In left-middle lane', 'In right-middle lane', 'In rightmost lane',
            # Ego motion state
            'Changing lane', 'Aligned with lane',
            # Front same-lane proximity
            'Front gap close', 'Front gap clear',
            # Rear same-lane proximity
            'Rear gap close', 'Rear gap clear',
            # Side feasibility (room to move)
            'Left lane free', 'Right lane free',
        ])

    # ---------- Public API (schema) ----------
    def state_to_binary(self, obs):
        """
        obs_5x5: np.ndarray of shape (5,5)
        Rows: vehicles; Cols: [x, y, vx, vy, heading]
        """ 
        obs_5x5 = obs.reshape((5, 5))
        ego, neighbors = self._split(obs_5x5)
        ego_lane = self._lane_index(ego[1]) 
        #print('ego lane:', ego_lane, 'ego y:', ego[1])
        lane_slots = self._lane_slots()

        b = [
            # 0–2 ego speed bands (mutually exclusive)
            self.ego_slow(ego),
            self.ego_medium(ego),
            self.ego_fast(ego),

            # 3–6 ego lane (mutually exclusive across 4 lanes)
            self.in_leftmost_lane(ego_lane),
            self.in_left_middle_lane(ego_lane, lane_slots),
            self.in_right_middle_lane(ego_lane, lane_slots),
            self.in_rightmost_lane(ego_lane),

            # 7–8 lateral motion state (mutually exclusive)
            self.changing_lane(ego),
            self.aligned_with_lane(ego),

            # 9–10 front same-lane
            self.front_gap_close(ego, neighbors, ego_lane),
            self.front_gap_clear(ego, neighbors, ego_lane),

            # 11–12 rear same-lane
            self.rear_gap_close(ego, neighbors, ego_lane),
            self.rear_gap_clear(ego, neighbors, ego_lane),

            # 13–14 side feasibility
            self.left_lane_free(ego, neighbors, ego_lane),
            self.right_lane_free(ego, neighbors, ego_lane),
        ]

        # Sanity: speed band XOR
        if sum(b[0:3]) != 1:
            # If borderline, force medium
            b[0], b[1], b[2] = 0, 1, 0

        # Sanity: lane bucket XOR (if lanes_count != 4 we still map to 4 slots best-effort)
        if sum(b[3:7]) != 1:
            # fallback: clamp into nearest valid slot
            slot = self._bucket_lane_into_4(ego_lane)
            b[3:7] = [0, 0, 0, 0]
            b[3 + slot] = 1

        # Sanity: lane motion XOR
        if b[7] + b[8] != 1:
            # default to aligned
            b[7], b[8] = 0, 1

        # Front/rear complements (don’t force XOR; unknown/empty also allowed)
        # But if neither set, prefer "clear"
        if b[9] + b[10] == 0:
            b[10] = 1
        if b[11] + b[12] == 0:
            b[12] = 1

        return np.array(b, dtype=int)

    def translate_state(self, binary_set):
        idx = np.where(binary_set == 1)[0]
        true_set = self.language_set[idx]
        if true_set.size == 0:
            return ''
        return ' and '.join(true_set.tolist())

    def feat_groups(self):
        # Group mutually exclusive or closely related features
        return [
            [0, 1, 2],        # speed bands
            [3, 4, 5, 6],     # lane slot
            [7, 8],           # lane-change vs aligned
            [9, 10],          # front gap
            [11, 12],         # rear gap
            [13],         # side feasibility
            [14],         # side feasibility
        ]

    def predicate_set(self):
        return [
            {'true': 'Ego slow', 'false': 'Not ego slow'},
            {'true': 'Ego medium', 'false': 'Not ego medium'},
            {'true': 'Ego fast', 'false': 'Not ego fast'},

            {'true': 'In leftmost lane', 'false': 'Not in leftmost lane'},
            {'true': 'In left-middle lane', 'false': 'Not in left-middle lane'},
            {'true': 'In right-middle lane', 'false': 'Not in right-middle lane'},
            {'true': 'In rightmost lane', 'false': 'Not in rightmost lane'},

            {'true': 'Changing lane', 'false': 'Not changing lane'},
            {'true': 'Aligned with lane', 'false': 'Not aligned with lane'},

            {'true': 'Front gap close', 'false': 'Front gap not close'},
            {'true': 'Front gap clear', 'false': 'Front gap not clear'},

            {'true': 'Rear gap close', 'false': 'Rear gap not close'},
            {'true': 'Rear gap clear', 'false': 'Rear gap not clear'},

            {'true': 'Left lane free', 'false': 'Left lane not free'},
            {'true': 'Right lane free', 'false': 'Right lane not free'},
        ]

    def num_predicates(self):
        return len(self.predicate_set())

    # ---------- Helpers ----------
    def _split(self, obs):
        """
        obs: array of shape (5, 5) with columns:
              [presence, x_norm, y, vx_norm, vy_norm]
        We drop the presence flag and keep [x, y, vx, vy].
        """
        obs = np.asarray(obs, dtype=float)
        # keep columns 1..4
        ego = obs[0, 1:]
        neighbors = obs[1:, 1:]
        return ego, neighbors


    def _lane_index(self, y):
        """
        Map lateral position y to a lane index in [0..lanes_count-1].
        Assumes lane centers at i*lane_width (or roughly so). Uses nearest lane.
        """
        lane = int(np.clip(round(y * (self.lanes_count - 1)), 0, self.lanes_count - 1))

        return lane

    def _lane_of_y(self, y):
        return self._lane_index(y)

    def _lane_slots(self):
        """
        Map actual lanes_count to 4 display buckets:
        0: leftmost, 1: left-middle, 2: right-middle, 3: rightmost
        For lanes_count != 4, we approximate:
        """
        # lane indices [0..L-1]; map to 4 buckets by normalized position
        return list(range(self.lanes_count))

    def _bucket_lane_into_4(self, lane):
        """
        Convert lane index [0..L-1] to 4 buckets.
        """
        L = self.lanes_count
        if L == 1:
            return 1  # treat as middle-left
        pos = lane / max(1, L - 1)
        if pos <= 0.0 + 1e-6: return 0    # leftmost
        if pos >= 1.0 - 1e-6: return 3    # rightmost
        # middle split
        return 1 if pos < 0.5 else 2

    def _same_lane(self, ego_y, other_y):
        # normalized lane spacing between adjacent lane centers
        lane_spacing = 1.0 / max(1, self.lanes_count - 1)
        lane_tol = lane_spacing * self.lane_tol_frac  # normalized tolerance
        return abs(other_y - ego_y) < lane_tol


    def _front_gap(self, ego, neighbors):
        ego_x, ego_y = ego[0], ego[1]
        front_dists = []
        for v in neighbors:
            dx_norm = v[0] - ego_x
            if dx_norm > 0 and self._same_lane(ego_y, v[1]):
                # convert to meters
                front_dists.append(dx_norm * self.x_span)
        return min(front_dists) if front_dists else np.inf

    def _rear_gap(self, ego, neighbors):
        ego_x, ego_y = ego[0], ego[1]
        rear_dists = []
        for v in neighbors:
            dx_norm = ego_x - v[0]
            if dx_norm > 0 and self._same_lane(ego_y, v[1]):
                rear_dists.append(dx_norm * self.x_span)
        return min(rear_dists) if rear_dists else np.inf


    def _lane_free(self, target_lane, ego, neighbors, forward_window=25.0, rear_window=15.0):
        if target_lane < 0 or target_lane >= self.lanes_count:
            return False
        ego_x = ego[0]
        for v in neighbors:
            v_lane = self._lane_of_y(v[1])
            if v_lane != target_lane:
                continue
            dx_norm = v[0] - ego_x
            dx = dx_norm * self.x_span
            if -rear_window <= dx <= forward_window:
                return False
        return True


    # ---------- Predicates ----------
    # Speed bands 
    def ego_slow(self, ego):
        vx_norm = ego[2] 
        return int(vx_norm < 0.334)  # 0.667 = slow_speed_max / fast_speed_min

    def ego_medium(self, ego):
        vx_norm = ego[2] 
        return int(0.334 <= vx_norm <  0.667)

    def ego_fast(self, ego):
        vx_norm = ego[2] 
        return int(vx_norm >= 667)


    # Lane slots (mapped to 4 canonical buckets)
    def in_leftmost_lane(self, lane):
        return int(self._bucket_lane_into_4(lane) == 0)

    def in_left_middle_lane(self, lane, _slots=None):
        return int(self._bucket_lane_into_4(lane) == 1)

    def in_right_middle_lane(self, lane, _slots=None):
        return int(self._bucket_lane_into_4(lane) == 2)

    def in_rightmost_lane(self, lane):
        return int(self._bucket_lane_into_4(lane) == 3)

    # Lateral motion
    def changing_lane(self, ego):
        vy = abs(ego[3])
        return int(vy > self.lane_change_vy)

    def aligned_with_lane(self, ego):
        vy = abs(ego[3])
        return int(vy <= self.lane_change_vy)

    # Front/rear proximity in current lane
    def front_gap_close(self, ego, neighbors, ego_lane):
        gap = self._front_gap(ego, neighbors)
        return int(gap < self.close_front_gap)

    def front_gap_clear(self, ego, neighbors, ego_lane):
        gap = self._front_gap(ego, neighbors)
        return int(gap >= self.close_front_gap)

    def rear_gap_close(self, ego, neighbors, ego_lane):
        gap = self._rear_gap(ego, neighbors)
        return int(gap < self.close_rear_gap)

    def rear_gap_clear(self, ego, neighbors, ego_lane):
        gap = self._rear_gap(ego, neighbors)
        return int(gap >= self.close_rear_gap)

    # Side feasibility
    def left_lane_free(self, ego, neighbors, ego_lane):
        target = ego_lane - 1
        return int(self._lane_free(target, ego, neighbors))

    def right_lane_free(self, ego, neighbors, ego_lane):
        target = ego_lane + 1
        return int(self._lane_free(target, ego, neighbors))
