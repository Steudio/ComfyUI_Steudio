import os
import sys

sys.path.append(os.path.dirname(__file__))
from _config_utils_ import _any_

class trigger_any:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": (_any_, {}),  # Any type will trigger
                "value": (_any_, {}),    # Any type to pass through unchanged
            }
        }

    RETURN_TYPES = (_any_,)  # Output is the same type as 'value'
    FUNCTION = "config"
    CATEGORY = "Steudio/Utils"

    def config(self, trigger, value):
        # Ignore 'trigger', just return 'value' unchanged
        return (value,)

NODE_CLASS_MAPPINGS = {"TriggerAny": trigger_any}
NODE_DISPLAY_NAME_MAPPINGS = {"TriggerAny": "Trigger Any"}