# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ******************************************  ShellClasses *************************************************************

"""Define globally-visible classes for Base classes and typechecking functions for parameters of PsyNeuLink Components

Shell Classes
-------------

Used to allow classes to refer to one another without creating import loops,
including (but not restricted to) the following dependencies:
- `Projection <Projection>` subclasses must see (particular) `State <State>` subclasses in order to assign
  `kwProjectionSender`
- `State <State>` subclasses must see (particular) `Projection <Projection>` subclasses in order to assign
  `PROJECTION_TYPE`
- `Process` must see `Mechanism <Mechanism>` subclasses to assign `PsyNeuLink.Components.DefaultMechanism`

TBI:
  `Mechanism <Mechanism>`, `Projection <Projection>` (and possibly `State <State>`) classes should be extensible:
  developers should be able to create, register and refer to subclasses (plug-ins), without modifying core code

"""

from psyneulink.components.component import Component


class ShellClassError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


def _attempt_to_call_base_class(cls, alternative):
    raise ShellClassError("Can't call {} directly;  must use {}".format(cls.__class__.__name__, alternative))


class ShellClass(Component):
    pass


# ******************************************* SYSTEM *******************************************************************

class System(ShellClass):

    def __init__(self,
                 default_variable=None,
                 size=None,
                 param_defaults=None,
                 name=None,
                 prefs=None,
                 context=None):
        if context is None:
            _attempt_to_call_base_class(self,'system()')
        super().__init__(default_variable=default_variable,
                         size=size,
                         param_defaults=param_defaults,
                         name=name,
                         prefs=prefs,
                         context=context)

    def execute(self, variable=None, time_scale=None, context=None):
        raise ShellClassError("Must implement execute in {0}".format(self.__class__.__name__))


# ****************************************** PROCESS *******************************************************************


class Process(ShellClass):
    def __init__(self,
                 default_variable=None,
                 size=None,
                 param_defaults=None,
                 name=None,
                 prefs=None,
                 context=None):
        if context is None:
            _attempt_to_call_base_class(self,'process()')
        super().__init__(default_variable=default_variable,
                         size=size,
                         param_defaults=param_defaults,
                         name=name,
                         prefs=prefs,
                         context=context)

# ******************************************* MECHANISM ****************************************************************


class Mechanism(ShellClass):

    def __init__(self,
                 default_variable=None,
                 size=None,
                 param_defaults=None,
                 name=None,
                 prefs=None,
                 context=None):
        if context is None:
            _attempt_to_call_base_class(self,'mechanism()')
        super().__init__(default_variable=default_variable,
                         size=size,
                         param_defaults=param_defaults,
                         name=name,
                         prefs=prefs,
                         context=context)

    def _validate_params(self, request_set, target_set=None, context=None):
        raise ShellClassError("Must implement _validate_params in {0}".format(self))

    def execute(self, variable, params, time_scale, context):
        raise ShellClassError("Must implement execute in {0}".format(self))

    def adjust_function(self, params, context):
        raise ShellClassError("Must implement adjust_function in {0}".format(self))


# ********************************************* STATE ******************************************************************


class State(ShellClass):

    @property
    def owner(self):
        raise ShellClassError("Must implement @property owner method in {0}".format(self.__class__.__name__))

    @owner.setter
    def owner(self, assignment):
        raise ShellClassError("Must implement @owner.setter method in {0}".format(self.__class__.__name__))

    @property
    def projections(self):
        raise ShellClassError("Must implement @property projections method in {0}".format(self.__class__.__name__))

    @projections.setter
    def projections(self, assignment):
        raise ShellClassError("Must implement @projections.setter method in {0}".format(self.__class__.__name__))

    def _validate_variable(self, variable, context=None):
        raise ShellClassError("Must implement _validate_variable in {0}".format(self))

    def _validate_params(self, request_set, target_set=None, context=None):
        raise ShellClassError("Must implement _validate_params in {0}".format(self))

    def add_observer_for_keypath(self, object, keypath):
        raise ShellClassError("Must implement add_observer_for_keypath in {0}".format(self.__class__.__name__))

    def set_value(self, new_value):
        raise ShellClassError("Must implement set_value in {0}".format(self.__class__.__name__))

    def update(self, params=None, time_scale=None, context=None):
        raise ShellClassError("{} must implement update".format(self.__class__.__name__))


# ******************************************* PROJECTION ***************************************************************


class Projection(ShellClass):

    # def assign_states(self):
    #     raise ShellClassError("Must implement assign_states in {0}".format(self.__class__.__name__))
    def validate_states(self):
        raise ShellClassError("Must implement validate_states in {0}".format(self.__class__.__name__))

    def _validate_params(self, request_set, target_set=None, context=None):
        raise ShellClassError("Must implement _validate_params in {0}".format(self.__class__.__name__))


# *********************************************  FUNCTION  *************************************************************


class Function(ShellClass):

    def execute(self, variable, params, context):
        raise ShellClassError("Must implement function in {0}".format(self))