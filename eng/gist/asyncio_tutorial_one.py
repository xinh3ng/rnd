"""
https://www.integralist.co.uk/posts/python-asyncio/
"""
import asyncio
import concurrent.futures
from random import randrange


# How to wait for multiple asynchronous tasks to complete.
if 1 == 0:

    async def foo(n):
        await asyncio.sleep(3 - n)  # wait 2s before continuing
        print(f"n: {n}!")

    async def main():
        tasks = [foo(1), foo(2), foo(3)]
        await asyncio.gather(*tasks)

    asyncio.run(main())


# Use the FIRST_COMPLETED option, meaning whichever task finishes first is what will be returned.
if 2 == 0:

    async def foo(n):
        s = randrange(2, 5)
        print(f"{n} will sleep for: {s} seconds")
        await asyncio.sleep(s)
        print(f"n: {n}!")

    async def main():
        tasks = [foo(1), foo(2), foo(3)]
        result = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        print(result)

    asyncio.run(main())


# How we can utilize a timeout to prevent waiting endlessly for an asynchronous task to finish.
if 3 == 0:

    async def foo(n):
        await asyncio.sleep(10)
        print(f"n: {n}!")

    async def main():
        try:
            await asyncio.wait_for(foo(1), timeout=5)
        except asyncio.TimeoutError:
            print("timeout!")

    asyncio.run(main())


# how to convert a coroutine into a Task and schedule it onto the event loop.
if 4 == 0:

    async def foo():
        await asyncio.sleep(10)
        print("Foo!")

    async def hello_world():
        task = asyncio.create_task(foo())
        print(task)

        await asyncio.sleep(5)
        print("Hello World!")

        await asyncio.sleep(10)
        print(task)

    asyncio.run(hello_world())


if 5 == 0:

    def blocking_io():
        # File operations (such as logging) can block the event loop: run them in a thread pool
        with open("/dev/urandom", "rb") as f:
            return f.read(1000)

    def cpu_bound():
        # CPU-bound operations will block the event loop: in general it is preferable to run them in a process pool
        return sum(i * i for i in range(3 * 10 ** 7))

    async def main():
        loop = asyncio.get_running_loop()

        # 1. Run in the default loop's executor:
        result = await loop.run_in_executor(None, blocking_io)
        print("default thread pool", result)

        # 2. Run in a custom thread pool:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            print("custom thread pool...")
            result = await loop.run_in_executor(pool, blocking_io)
            print("custom thread pool result:", result)

        # 3. Run in a custom process pool:
        with concurrent.futures.ProcessPoolExecutor() as pool:
            print("custom process pool...")
            result = await loop.run_in_executor(pool, cpu_bound)
            print("custom process pool result:", result)

    asyncio.run(main())

#
print("ALL DONE!")
