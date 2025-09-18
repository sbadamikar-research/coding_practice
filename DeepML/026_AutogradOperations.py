class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        result = Value((self.data + other.data), [self, other], '+')
        
        def backward_fn():
            self.grad += result.grad
            other.grad += result.grad 
            
        result._backward = backward_fn
        return result

    def __mul__(self, other):
        result = Value((self.data * other.data), [self, other], '*')
        
        def backward_fn():            
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad
        
        result._backward = backward_fn
        return result
    
    def relu(self):
        result = Value(max(0, self.data), [self,], 'ReLU')
        
        def backward_fn():
            self.grad += (result.data > 0) * result.grad
            
        result._backward = backward_fn
        return result
    
    def backward(self):
        self.grad = 1
        
        backtrace = [self]
        visited = []
        for val in backtrace:
            # print("Initial: ", val)
            if (val in visited):
                continue
            
            for child in val._prev:
                # print("appending ", child.data)
                backtrace.append(child)
                
            # print("Calling ", val.data)
            val._backward()
            visited.append(val)
            # print("Updated: ", val)
        
        
a = Value(2)
b = Value(-3)
c = Value(10)
d = a + b * c
e = d.relu()
e.backward()
print(a, b, c, d, e)