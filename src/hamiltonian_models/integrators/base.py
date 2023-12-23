'''Base class for integrators

How to add an integrator:
    1.) add its implementation in the integrators folder with base class |Integrator|
    2.) add the integrator to the available_integrators attribute in |DynamicalSystem|'''
from abc import ABC, abstractmethod

class Integrator(object):
    '''A base class for integrators.'''
    def __init__(self, verbose):
        self.verbose = verbose
        self.print_progress_frequency = 0.1 # output progress every 10 percent

    @staticmethod
    @abstractmethod
    def is_applicable(dyn_sys):
        '''Checks, if the integrator is applicable to the given |DynamicalSystem| supporting the duck-typing principle.
        Parameters
        ----------
        dyn_sys
            the |DynamicalSystem| to check
        
        Returns
        -------
        boolean, if the integrator is applicable to obj'''
        pass

    @property
    @abstractmethod
    def is_explicit(self):
        pass

    @abstractmethod
    def solve(self, dyn_sys, mu, t_0, t_end, hook_fcns=()):
        '''Solves the given |DynamicalSystem| with the

        Parameters
        ----------
        dyn_sys
            The |DynamicalSystem| to solve.
        mu
            The parameter set.
        t_0
            The inital time.
        t_end
            The final time.
        hook_fcns
            An |Iterable| containing multiple instances of |HookFunction| or a single |HookFunction| which
            are/is setup at the beginning of the method and evaluated before and after each completed time step.

        Returns
        -------
        A |TimeDataList| with the discrete time-steps and the solution as data.
        '''
        assert self.__class__.is_applicable(dyn_sys)
        pass

    def print_progress(self, last, current, total, iter_info=''):
        '''Prints progress and given iter_info in intervals specified by self.print_progress_frequency.
        It should hold
            0 <= last <= current <= total
        Parameters
        ----------
        last
            last iteration
        current
            current iteration
        total
            total iterations
        iter_info
            additional information to be printed.'''
        idx_last = int((last/total)/self.print_progress_frequency)
        idx_current = int((current/total)/self.print_progress_frequency)
        if self.verbose and idx_last != idx_current:
            print(str(self.__class__.__name__) + ': %3d%%' % (current/total*100) + iter_info)


class FixedTimeStepWidthIntegrator(Integrator):
    '''Integrator with fixed time-step width.'''
    def __init__(self, dt, verbose=True):
        '''
        Parameters
        ----------
        dt
            The fixed time-step width.
        '''
        self._dt = dt
        super(FixedTimeStepWidthIntegrator, self).__init__(verbose)

    def __str__(self):
        return self.__class__.__name__ + '({:4.2e})'.format(self._dt)

    def __repr__(self):
        return self.__str__()


class TimeDataList(object):
    '''A list to hold a pair (td_t, td_data) of lists consisting of
        td_t:    the corresponding time instances and
        td_data: some data.
    '''
    def __init__(self, td_t=None, td_data=None):
        if td_t is not None and td_data is not None:
            assert len(td_t) == len(td_data)
            self._t = td_t
            self._data = td_data
        elif not td_t and not td_data:
            self._t = []
            self._data = []
        else:
            raise RuntimeError('Please, specify both, td_t and td_data, or neither of both.')

    def append(self, t, data):
        self._t.append(t)
        self._data.append(data)

    def all_data(self):
        assert len(self._data) == len(self._t)
        return self._data

    def all_t(self):
        assert len(self._data) == len(self._t)
        return self._t

    def pop(self):
        return self._t.pop(), self._data.pop()

    def __getitem__(self, i):
        assert len(self._data) == len(self._t)
        if isinstance(i, int):
            assert -len(self._t) <= i < len(self._t)
            return self._t[i], self._data[i]

        elif isinstance(i, slice):
            assert all(bound is None or -len(self._t) <= bound < len(self._t) for bound in [i.start, i.stop])
            sliced_td_list = TimeDataList()
            sliced_td_list._t = self._t[i]
            sliced_td_list._data = self._data[i]
            return sliced_td_list

        else:
            raise RuntimeError('Unknown data type for i: ', type(i))

    def __iter__(self):
        assert len(self._data) == len(self._t)
        for i in range(len(self._data)):
            yield self._t[i], self._data[i]

    def __len__(self):
        len_data = len(self._data)
        assert len_data == len(self._t)
        return len_data


class HookFunction(ABC):
    '''A custom |HookFunction| that is evaluated during the solve method of the |Integrator| after each comleted time step.
    The output is supposed to be stored inside the instance of |HookFunction| itself.
    '''
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def setup(self, x0, mu, nt):
        pass

    @abstractmethod
    def eval(self, *arg, **kwarg):
        pass

    @abstractmethod
    def finalize(self, *arg, **kwarg):
        pass
