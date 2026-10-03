# Sample states for testing the state encoding and reward functions in the MORL-APG environment.

################################################################################################
# sample states for testing the state encoding and reward functions in the multi-objective fruittree environment

test_states_ft = [
        {"lvl": 0,"pos": 0},
        {"lvl": 1,"pos": 0},
        {"lvl": 1,"pos": 1},
        {"lvl": 2,"pos": 1},
        {"lvl": 2,"pos": 2},
        {"lvl": 3,"pos": 1},
        {"lvl": 3,"pos": 2},
        {"lvl": 3,"pos": 3},
        ]

################################################################################################
# sample states for testing the state encoding and reward functions in the multi-objective highway environment

test_states_hw = [
        # centered lane, medium speed, traffic ahead
        {
            "xEgo": 0.85,
            "yEgo": 0.33,
            "vxEgo": 0.30,
            "vyEgo": 0.0,
            "xC1": 0.10,
            "yC1": 0.33,
        },

        # upper lane
        {
            "xEgo": 0.90,
            "yEgo": 0.66,
            "vxEgo": 0.36,
            "vyEgo": 0.0,
            "xC1": 0.05,
            "yC1": 0.66,
        },

        # lower lane
        {
            "xEgo": 0.88,
            "yEgo": 0.0,
            "vxEgo": 0.28,
            "vyEgo": 0.0,
            "xC1": 0.12,
            "yC1": 0.0,
        },

        # faster ego vehicle
        {
            "xEgo": 0.95,
            "yEgo": 0.33,
            "vxEgo": 0.375,
            "vyEgo": 0.0,
            "xC1": 0.20,
            "yC1": 0.33,
        },

        # slower traffic nearby
        {
            "xEgo": 0.80,
            "yEgo": 0.33,
            "vxEgo": 0.25,
            "vyEgo": 0.0,
            "xC1": -0.02,
            "yC1": 0.33,
        },

        # lane change scenario
        {
            "xEgo": 0.87,
            "yEgo": 0.50,
            "vxEgo": 0.31,
            "vyEgo": 0.01,
            "xC1": 0.08,
            "yC1": 0.66,
        },

        # dense traffic
        {
            "xEgo": 0.84,
            "yEgo": 0.33,
            "vxEgo": 0.29,
            "vyEgo": 0.0,
            "xC1": 0.05,
            "yC1": 0.33,
            "xC2": 0.15,
            "yC2": 0.66,
        },

        # more open road
        {
            "xEgo": 0.92,
            "yEgo": 0.33,
            "vxEgo": 0.37,
            "vyEgo": 0.0,
            "xC1": 0.25,
            "yC1": 0.66,
        },
        # balanced medium-density traffic
        {
            "xEgo": 0.86,
            "yEgo": 0.33,
            "vxEgo": 0.31,
            "vyEgo": 0.0,

            "xC1": 0.10,
            "yC1": 0.33,
            "vxC1": -0.05,

            "xC2": 0.22,
            "yC2": 0.66,
            "vxC2": -0.03,

            "xC3": 0.35,
            "yC3": 0.0,
            "vxC3": 0.01,
        },

        # ego boxed in between cars
        {
            "xEgo": 0.84,
            "yEgo": 0.33,
            "vxEgo": 0.28,
            "vyEgo": 0.0,

            "xC1": 0.05,
            "yC1": 0.33,
            "vxC1": -0.10,

            "xC2": 0.08,
            "yC2": 0.66,
            "vxC2": -0.07,

            "xC3": 0.06,
            "yC3": 0.0,
            "vxC3": -0.08,

            "xC4": 0.18,
            "yC4": 0.66,
            "vxC4": 0.00,
        },

        # open left lane
        {
            "xEgo": 0.91,
            "yEgo": 0.66,
            "vxEgo": 0.37,
            "vyEgo": 0.0,

            "xC1": 0.18,
            "yC1": 0.66,
            "vxC1": -0.02,

            "xC2": 0.32,
            "yC2": 0.33,
            "vxC2": 0.01,

            "xC3": 0.48,
            "yC3": 0.0,
            "vxC3": 0.02,
        },

        # congested lower lane
        {
            "xEgo": 0.82,
            "yEgo": 0.0,
            "vxEgo": 0.26,
            "vyEgo": 0.0,

            "xC1": 0.02,
            "yC1": 0.0,
            "vxC1": -0.15,

            "xC2": 0.11,
            "yC2": 0.33,
            "vxC2": -0.08,

            "xC3": 0.20,
            "yC3": 0.66,
            "vxC3": -0.04,

            "xC4": 0.28,
            "yC4": 0.0,
            "vxC4": -0.01,
        },

        # aggressive fast traffic
        {
            "xEgo": 0.95,
            "yEgo": 0.33,
            "vxEgo": 0.375,
            "vyEgo": 0.0,

            "xC1": 0.07,
            "yC1": 0.33,
            "vxC1": -0.18,

            "xC2": 0.14,
            "yC2": 0.66,
            "vxC2": -0.16,

            "xC3": 0.26,
            "yC3": 0.0,
            "vxC3": -0.14,

            "xC4": 0.41,
            "yC4": 0.66,
            "vxC4": -0.10,
        },

        # nearly empty highway
        {
            "xEgo": 0.93,
            "yEgo": 0.33,
            "vxEgo": 0.36,
            "vyEgo": 0.0,

            "xC1": 0.40,
            "yC1": 0.66,
            "vxC1": 0.01,

            "xC2": 0.55,
            "yC2": 0.0,
            "vxC2": 0.02,
        },

        # active lane change upward
        {
            "xEgo": 0.88,
            "yEgo": 0.50,
            "vxEgo": 0.32,
            "vyEgo": 0.014,

            "xC1": 0.09,
            "yC1": 0.66,
            "vxC1": -0.04,

            "xC2": 0.16,
            "yC2": 0.33,
            "vxC2": -0.02,

            "xC3": 0.31,
            "yC3": 0.0,
            "vxC3": 0.01,
        },

        # active lane change downward
        {
            "xEgo": 0.87,
            "yEgo": 0.18,
            "vxEgo": 0.31,
            "vyEgo": -0.014,

            "xC1": 0.05,
            "yC1": 0.0,
            "vxC1": -0.07,

            "xC2": 0.12,
            "yC2": 0.33,
            "vxC2": -0.05,

            "xC3": 0.24,
            "yC3": 0.66,
            "vxC3": -0.01,
        },

        # surrounded but moving faster than traffic
        {
            "xEgo": 0.94,
            "yEgo": 0.33,
            "vxEgo": 0.37,
            "vyEgo": 0.0,

            "xC1": 0.03,
            "yC1": 0.33,
            "vxC1": -0.20,

            "xC2": 0.06,
            "yC2": 0.66,
            "vxC2": -0.18,

            "xC3": 0.08,
            "yC3": 0.0,
            "vxC3": -0.17,

            "xC4": 0.15,
            "yC4": 0.66,
            "vxC4": -0.11,
        },

        # unstable mixed-speed traffic
        {
            "xEgo": 0.85,
            "yEgo": 0.33,
            "vxEgo": 0.29,
            "vyEgo": 0.0,

            "xC1": -0.01,
            "yC1": 0.33,
            "vxC1": -0.21,

            "xC2": 0.19,
            "yC2": 0.66,
            "vxC2": 0.02,

            "xC3": 0.27,
            "yC3": 0.0,
            "vxC3": -0.12,

            "xC4": 0.38,
            "yC4": 0.66,
            "vxC4": 0.01,
        },
] 

################################################################################################
# sample states for testing the state encoding and reward functions in the multi-objective deep sea treasure environment

test_states_dst = [
        {"x": 0, "y": 0},  # start
        {"x": 1, "y": 1},
        {"x": 2, "y": 2},
        {"x": 3, "y": 3},
        {"x": 4, "y": 4},
        {"x": 5, "y": 5},
        {"x": 6, "y": 7}, 
        {"x": 10, "y": 10},
]
