Quickstart Example
##################

This page provides a short example of how
:doc:`alogos <package_references>`
can be used to tackle a simple optimization problem, the search for
an arithmetic expression that approximates the number
`π <https://en.wikipedia.org/wiki/Pi>`__.
It assumes that you are using a Python 3.6+ interpreter and
that you have installed alogos as explained in the
:doc:`Installation Guide <installation>`::

    import alogos as al

    bnf_text = """
    <S> ::= <NUM> <OP> <S> | <NUM> <OP> <NUM>
    <NUM> ::= <DIGIT> | <DIGIT> <DIGIT>
    <OP> ::= + | * | - | /
    <DIGIT> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
    """
    grammar = al.Grammar(bnf_text)

    def objective_function(string):
        pi = 3.141592653589793
        number = eval(string)
        return abs(number-pi)

    ea = al.EvolutionaryAlgorithm(grammar, objective_function, 'min', verbose=True, max_generations=300)
    best_ind = ea.run()
    print()
    print('Solution:', best_ind.phenotype)



This code uses three main steps to define and solve an optimization
problem:

1. A **search space** is defined by providing a grammar. A grammar is
   a device to define a formal language, which is simply a set of
   strings.

   In this case, the grammar's language is an infinite set of
   strings, each representing an arithmetic expression. While this
   might sound complicated, it just means that the search space
   consists of strings such as ``"22*2/18+39"``, ``"11*5-0"``,
   ``"36/85-7"`` and an unbounded amount of other possible combinations
   of numbers and operators. There is an implicit bias towards shorter
   strings, so expressions with thousands of numbers and operators are
   part of the search space but unlikely to be actually visited by the
   search algorithm in a practical amount of time.
2. A **search goal** is defined by implementing an objective function
   that rates the quality of each candidate solution in the search
   space. It gets a string as input and returns a numerical value for it
   as output, which represents a quality score for the given string.

   In this case, each string that the function gets is an arithmetic
   expression. Therefore the string can be evaluated to a number 
   (e.g. ``eval("2+17/91*15")`` gives ``4.802197802197802``)
   and then it can be determined how close this number is to pi.
   The smaller the difference between the number and pi is (e.g.
   ``abs(4.802197802197802-3.141592653589793)`` gives
   ``1.660605148608009``), the better the arithmetic expression
   approximates pi.
3. A **search method** is defined by creating an instance of a chosen
   algorithm and passing search space and search goal to it in form of
   a) the grammar, b) the objective function and c) an objective which
   can be minimization (``"min"``) or maximization (``"max"``) of the
   value provided by the objective function.
   The task of the algorithm is then to look for an optimal solution
   among all the candidate solutions in the search space.
   Ideally it will find a good solution after evaluating only a small
   number of candidates.

   In this case, an evolutionary algorithm is used that searches
   for an arithmetic expression, which gets a minimal value assigned
   by the objective function and therefore is a good approximation
   of pi.



Executing the code will generate an output that looks similar but not
identical to this one, because the evolutionary algorithm is stochastic
and therefore creates a different output each time::

    Progress         Generations      Evaluations      Runtime (sec)    Best fitness    
    ..... .....      10               992              0.6              0.01969766899085279
    ..... .....      20               1957             0.8              0.0012644892673496777
    ..... .....      30               2764             0.9              0.0012644892673496777
    ..... .....      40               3653             1.1              0.0002883057637053099
    ..... .....      50               4490             1.2              0.0002883057637053099
    ..... .....      60               5416             1.4              0.0002883057637053099
    ..... .....      70               6359             1.5              0.0002883057637053099
    ..... .....      80               7318             1.6              0.0002883057637053099
    ..... .....      90               8264             1.7              0.0002883057637053099
    ..... .....      100              9204             1.9              0.0002883057637053099
    ..... .....      110              10136            2.0              0.0002883057637053099
    ..... .....      120              11077            2.1              0.0002883057637053099
    ..... .....      130              12032            2.2              0.0002883057637053099
    ..... .....      140              12982            2.3              0.0002883057637053099
    ..... .....      150              13926            2.4              0.0002883057637053099
    ..... .....      160              14886            2.5              0.0002883057637053099
    ..... .....      170              15875            2.6              0.0002883057637053099
    ..... .....      180              16862            2.7              7.37726408894801e-06
    ..... .....      190              17839            2.9              2.0435200820401178e-07
    ..... .....      200              18802            3.0              2.0435200820401178e-07
    ..... .....      210              19771            3.2              2.0435200820401178e-07
    ..... .....      220              20720            3.4              2.0435200820401178e-07
    ..... .....      230              21693            3.6              2.0435200820401178e-07
    ..... .....      240              22652            3.7              2.0435200820401178e-07
    ..... .....      250              23611            3.9              2.0435200820401178e-07
    ..... .....      260              24561            4.0              2.0435200820401178e-07
    ..... .....      270              25531            4.1              2.0435200820401178e-07
    ..... .....      280              26496            4.2              2.0435200820401178e-07
    ..... .....      290              27460            4.3              2.0435200820401178e-07
    ..... .....      300              28410            4.5              2.0435200820401178e-07

    Finished         300              28410            4.5              2.0435200820401178e-07

    Solution: 13/92-9+6+6+1/5*1/89*5/39



The five columns have following meaning:

1. Each dot represents one generation in the evolutionary search and is
   printed when evaluating the individuals in it is finished. If the
   objective function is demanding this can take quite a while, but here
   each generation is generated and evaluated in a fraction of seconds.
2. Whenever 10 generations are completed, the total number of
   generations so far is shown.
3. The objective function is evaluated for an increasing number of
   candidate solutions throughout the run. By default, duplicate
   individuals are evaluated only once throughout the run to prevent
   potentially costly recalculations.
4. Depending on the objective function, the evaluation can be time
   consuming, so it is good to keep a track of total time passed.
5. The goal is to find a candidate solution with a good value assigned
   by the objective function, which is also known as its fitness, and
   ideally it should improve throughout the run. In this particular
   case, the fitness value means how far the best expression found so
   far deviates from pi.

The last row shows the best solution found in this run:

- The best arithmetic expression that was discovered by creating 300
  consecutive generations within 4.5 seconds is
  ``"13/92-9+6+6+1/5*1/89*5/39"``.
- Evaluating this expression results in the number
  ``3.141592449237785``, which correctly approximates
  ``pi=3.141592653589793`` in the first six digits after the comma.
- If the algorithm were allowed to search for a longer time, the result
  would most likely have continued to improve, but there is never a
  guarantee because the search is stochastic and might get trapped in
  a bad region of the search space for a while. For this reason,
  it can sometimes be better to start a fresh run rather than continuing
  a stuck one. Increasing the population size and tweaking other
  parameters can also reduce the chance of getting stuck, which is less
  important for toy problems such as the one shown here, but becomes
  relevant when dealing with hard real-world problems.



Here is the output of a longer run that used ``max_generations=500``.
It actually resulted in an arithmetic expression that approximates pi
to the full precision it was provided::

    Progress         Generations      Evaluations      Runtime (sec)    Best fitness    
    ..... .....      10               987              0.8              0.006977268974408535
    ..... .....      20               1953             0.9              0.006977268974408535
    ..... .....      30               2912             1.0              0.0004636213317286142
    ..... .....      40               3871             1.1              0.0004636213317286142
    ..... .....      50               4841             1.2              0.0004636213317286142
    ..... .....      60               5825             1.4              0.0004636213317286142
    ..... .....      70               6815             1.4              0.0004636213317286142
    ..... .....      80               7798             1.5              0.00040724325544694295
    ..... .....      90               8778             1.6              0.00040724325544694295
    ..... .....      100              9744             1.7              0.0003328122487156193
    ..... .....      110              10708            1.8              0.0003328122487156193
    ..... .....      120              11678            1.9              0.0003328122487156193
    ..... .....      130              12636            2.0              0.0003034549835216893
    ..... .....      140              13604            2.1              0.0002883057637061981
    ..... .....      150              14555            2.2              0.0002883057637061981
    ..... .....      160              15525            2.4              0.0002883057637061981
    ..... .....      170              16496            2.5              0.0002883057637061981
    ..... .....      180              17462            2.6              0.0002883057637061981
    ..... .....      190              18424            2.7              1.3094937064472845e-05
    ..... .....      200              19400            2.8              8.682170400398093e-06
    ..... .....      210              20373            2.9              1.7688359603695858e-06
    ..... .....      220              21350            3.1              5.590024336754595e-07
    ..... .....      230              22328            3.3              2.351036076930768e-08
    ..... .....      240              23313            3.5              2.351000460976138e-08
    ..... .....      250              24306            3.7              2.01513463693459e-08
    ..... .....      260              25291            3.9              4.78525308267308e-09
    ..... .....      270              26285            4.2              2.2676083233363897e-11
    ..... .....      280              27263            4.6              2.2672530519685097e-11
    ..... .....      290              28247            5.0              2.2216894990378933e-11
    ..... .....      300              29238            5.4              1.0601297617540695e-11
    ..... .....      310              30217            5.8              1.9761969838327786e-12
    ..... .....      320              31204            6.2              7.416289804496046e-13
    ..... .....      330              32183            6.5              7.416289804496046e-13
    ..... .....      340              33170            7.0              7.416289804496046e-13
    ..... .....      350              34153            7.3              1.8740564655672642e-13
    ..... .....      360              35128            7.7              1.8740564655672642e-13
    ..... .....      370              36109            8.3              3.2862601528904634e-14
    ..... .....      380              37098            8.8              1.1546319456101628e-14
    ..... .....      390              38075            9.3              1.1546319456101628e-14
    ..... .....      400              39049            9.8              7.993605777301127e-15
    ..... .....      410              40043            10.2             7.993605777301127e-15
    ..... .....      420              41028            10.6             7.993605777301127e-15
    ..... .....      430              42011            11.0             7.993605777301127e-15
    ..... .....      440              42987            11.4             0.0
    ..... .....      450              43973            11.8             0.0
    ..... .....      460              44959            12.2             0.0
    ..... .....      470              45940            12.7             0.0
    ..... .....      480              46920            13.1             0.0
    ..... .....      490              47903            13.5             0.0
    ..... .....      500              48889            14.0             0.0

    Finished         500              48889            14.0             0.0

    Solution: 00-5+3/4/7+7/47/3/28/7-7/7/40/7/43/3/24/42/7/17/3/28/7-7/7/40/7/41/3/24/47/3-46/7/7/47/7/41/3/86/47/3/46/41/7/20-47/9/16/9/76/71/7/42/23/71/13/16/7/47/3*28/4-7/7/40/7/41/3/24/47/3-16/7/7/10/7/21/3/28/47/3*46/41/7+20/5/3/16/7/7/20/7/41/2/71/3/27*43/3/16/7*7-40/7/41/3/28/47/3/46/76+7/7/41/3/27/6+7/41/5+5*0+8


- The best arithmetic expression that was discovered by creating 500
  consecutive generations within 14.0 seconds is
  ``"00-5+3/4/7+7/47/3/28/7-7/7/40/7/43/3/24/42/7/17/3/28/7-7/7/40/7/41/3/24/47/3-46/7/7/47/7/41/3/86/47/3/46/41/7/20-47/9/16/9/76/71/7/42/23/71/13/16/7/47/3*28/4-7/7/40/7/41/3/24/47/3-16/7/7/10/7/21/3/28/47/3*46/41/7+20/5/3/16/7/7/20/7/41/2/71/3/27*43/3/16/7*7-40/7/41/3/28/47/3/46/76+7/7/41/3/27/6+7/41/5+5*0+8"``.
- Evaluating this expression results in the number
  ``3.141592653589793``, which correctly approximates
  ``pi=3.141592653589793`` in all 15 digits after the comma that
  were provided.

Exercise for the reader:

- To achieve even higher precision, Python's built-in module
  `decimal <https://docs.python.org/3/library/decimal.html>`__
  could be used to overcome the limitations of
  `floating point numbers
  <https://docs.python.org/3/library/stdtypes.html#typesnumeric>`__
  and
  `floating point arithmetic
  <https://docs.python.org/3/tutorial/floatingpoint.html>`__.
- To bias the search towards shorter arithmetic expressions,
  the objective function could be modified so that the length of a
  candidate string is taken into consideration by the fitness
  calculation in such a way that shorter expressions get a slightly
  better value.
