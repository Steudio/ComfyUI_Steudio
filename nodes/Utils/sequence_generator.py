
# sequence_generator.py
# Sequence_Generator utilize code from Cubiq    https://github.com/cubiq/ComfyUI_essentials

class Sequence_Generator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gen": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "0...1+0.1"}),
            }
        }
    
    RETURN_TYPES = ("INT", "FLOAT", )
    OUTPUT_IS_LIST = (True,True)
    OUTPUT_NODE = True
    FUNCTION = "Execute"
    CATEGORY = "Steudio/Utils"
    DESCRIPTION = """
x...y+z | Generates a sequence of numbers from x to y with a step of z.
x...y#z | Generates z evenly spaced numbers between x and y.
  x,y,z | Generates a list of x, y, z.
    """

    def Execute(self, gen):
        elements = gen.split(',')
        result = []

        def parse_number(s):
            try:
                return float(s)
            except ValueError:
                return 0.0

        for element in elements:
            element = element.strip()

            if '...' in element:
                if '#' in element:
                    start, rest = element.split('...')
                    end, num_items = rest.split('#')
                    start = parse_number(start)
                    end = parse_number(end)
                    num_items = int(parse_number(num_items))
                    if num_items == 1:
                        result.append(round(start, 2))
                    else:
                        step = (end - start) / (num_items - 1)
                        for i in range(num_items):
                            result.append(round(start + i * step, 2))
                else:
                    start, rest = element.split('...')
                    end, step = rest.split('+')
                    start = parse_number(start)
                    end = parse_number(end)
                    step = abs(parse_number(step))
                    current = start
                    if start > end:
                        step = -step
                    while (step > 0 and current <= end) or (step < 0 and current >= end):
                        result.append(round(current, 2))
                        current += step
            else:
                result.append(round(parse_number(element), 2))

        seq_int = list(map(int, result))
        seq_float = list(map(float, [f"{num:.2f}"for num in result if isinstance(num, float)]))
        seq_int_float = f"{len(seq_int)} INT: {seq_int}\n{len(seq_float)} FLOAT: {seq_float}"

        return {"ui": {"text": (seq_int_float)}, "result": (seq_int, seq_float)}
    

NODE_CLASS_MAPPINGS = {
    "Sequence Generator": Sequence_Generator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Sequence Generator": "Sequence Generator",
}
