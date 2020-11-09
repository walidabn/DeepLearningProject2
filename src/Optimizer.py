class Optimizer(object):
    """ Optimizer is a skeleton that should be used for implementing various optimization algorithms, such as Adam or RMSProp.
    """
    
    def step(self,*input):
        """ All optimizers implement a step() method, that updates the parameters depending on the optimization procedure.
        """
        raise NotImplementedError
        
    def param(self):
        """ This method should either return the parameters of the optimizers, if it has some, or the empty list otherwise.
        """
        return []