import multiprocessing as mp
import time

def parallel_optimize(self, iterations):
        """
        The `parallel_optimize` function uses multiprocessing to run optimization iterations in parallel
        and selects the best solution based on fitness evaluation.

        :param iterations: The `iterations` parameter in the `parallel_optimize` method represents the
        number of iterations or steps that the optimization algorithm will take to find the best
        solution. It determines how many times the optimization process will be repeated to improve the
        solution. The higher the number of iterations, the more chances the algorithm 
        """
        threshold = 50  # Set your threshold value here

        if (
            len(self.courses) > threshold
        ):  # Add a condition to check the size of your dataset
            num_processes = mp.cpu_count()
            pool = mp.Pool(processes=num_processes)

            start_time = time.time()

            results = [
                pool.apply_async(self.optimize, args=(iterations // num_processes,))
                for _ in range(num_processes)
            ]
            pool.close()
            pool.join()

            end_time = time.time()
            self.woa_time = end_time - start_time

            solutions = [result.get() for result in results if result.get() is not None]
            
            if solutions:
                self.best_solution = min(
                    solutions, key=lambda x: self.evaluate_fitness(x)
                )
        else:
            start_time = time.time()
            self.best_solution = self.optimize(iterations)
            end_time = time.time()
            self.woa_time = end_time - start_time

        return self.best_solution