import asyncio

async def f():
    #future.set_result(0)
    return 0

async def main():
    print(await asyncio.gather(f(), f(), f()))

asyncio.run(main())
#futures = [f() for x in range(3)]
#loop = asyncio.get_event_loop()
#loop.run_until_complete(asyncio.wait(futures))
#print([x.result() for x in futures])
