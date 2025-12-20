#!/usr/bin/env python3
"""
PicoScope Performance Benchmark Script

This script helps you test different PicoScope sampling modes to compare performance.
It measures sampling rates, latency, and throughput for each mode.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import ctypes
from picosdk.ps3000a import ps3000a as ps
from picosdk.functions import adc2mV, mV2adc, assert_pico_ok

class PicoScopePerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.sampling_rates = []
        self.latencies = []
        self.throughputs = []
        
    def get_fastest_timebase(self, chandle):
        """Find the fastest supported timebase for this PicoScope model"""
        # Try timebases starting from 0 until we find one that works
        for timebase in range(0, 10):  # Try timebases 0-9
            try:
                timeIntervalns = ctypes.c_float()
                returnedMaxSamples = ctypes.c_int16()
                
                # Test if this timebase works with a small sample count
                test_samples = 1000
                status = ps.ps3000aGetTimebase2(chandle, 
                                                timebase, 
                                                test_samples, 
                                                ctypes.byref(timeIntervalns), 
                                                1, 
                                                ctypes.byref(returnedMaxSamples), 
                                                0)
                
                if status == 0:  # PICO_OK
                    print(f"Fastest supported timebase: {timebase} (interval: {timeIntervalns.value:.2f} ns)")
                    return timebase
                    
            except Exception as e:
                continue
        
        # If no timebase works, default to 2 (known to work from benchmark)
        print("Using default timebase: 2")
        return 2

    def benchmark_block_mode(self, timebase=2, samples=30000, num_buffers=20, duration=10):
        """Benchmark standard block mode"""
        print(f"\n=== Benchmarking Block Mode ===")
        print(f"Timebase: {timebase}, Samples: {samples}, Buffers: {num_buffers}")
        
        # Setup PicoScope
        status = {}
        chandle = ctypes.c_int16()
        
        # Open unit
        status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)
        
        try:
            assert_pico_ok(status["openunit"])
        except:
            powerstate = status["openunit"]
            if powerstate == 282:
                status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
            elif powerstate == 286:
                status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
            else:
                raise
            assert_pico_ok(status["ChangePowerSource"])
        
        # Set channel A
        chARange = 6
        status["setChA"] = ps.ps3000aSetChannel(chandle, 0, 1, 1, chARange, 0)
        assert_pico_ok(status["setChA"])
        
        # Get max ADC value
        maxADC = ctypes.c_int16()
        status["maximumValue"] = ps.ps3000aMaximumValue(chandle, ctypes.byref(maxADC))
        assert_pico_ok(status["maximumValue"])
        
        # If timebase is 0, find the fastest supported timebase
        if timebase == 0:
            timebase = self.get_fastest_timebase(chandle)
        
        # Setup trigger
        adcTriggerLevel = mV2adc(500, chARange, maxADC)
        channelProperties = ps.PS3000A_TRIGGER_CHANNEL_PROPERTIES(
            adcTriggerLevel, 10, adcTriggerLevel, 10,
            ps.PS3000A_CHANNEL["PS3000A_CHANNEL_A"],
            ps.PS3000A_THRESHOLD_MODE["PS3000A_LEVEL"])
        
        status["setTrigProp"] = ps.ps3000aSetTriggerChannelProperties(
            chandle, ctypes.byref(channelProperties), 1, 0, 10000)
        assert_pico_ok(status["setTrigProp"])
        
        # Setup memory segments
        cmaxSamples = ctypes.c_int32(samples)
        status["MemorySegments"] = ps.ps3000aMemorySegments(chandle, num_buffers, ctypes.byref(cmaxSamples))
        assert_pico_ok(status["MemorySegments"])
        
        status["SetNoOfCaptures"] = ps.ps3000aSetNoOfCaptures(chandle, num_buffers)
        assert_pico_ok(status["SetNoOfCaptures"])
        
        # Setup data buffers
        buffers = []
        for i in range(num_buffers):
            buffer_max = np.empty(samples, dtype=np.int16)
            buffer_min = np.empty(samples, dtype=np.int16)
            buffers.append((buffer_max, buffer_min))
            
            status[f"SetDataBuffers_{i}"] = ps.ps3000aSetDataBuffers(
                chandle, 0, buffer_max.ctypes.data, buffer_min.ctypes.data, samples, i, 0)
            assert_pico_ok(status[f"SetDataBuffers_{i}"])
        
        # Benchmark
        start_time = time.time()
        acquisition_times = []
        acquisition_count = 0
        
        print("Starting benchmark...")
        
        while time.time() - start_time < duration:
            # Time individual acquisition
            acq_start = time.time()
            
            # Run block
            status["runblock"] = ps.ps3000aRunBlock(chandle, 0, samples, timebase, 1, None, 0, None, None)
            assert_pico_ok(status["runblock"])
            
            # Wait for completion
            ready = ctypes.c_int16(0)
            check = ctypes.c_int16(0)
            while ready.value == check.value:
                status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))
            
            # Get bulk data
            overflow = (ctypes.c_int16 * num_buffers)()
            status["GetValuesBulk"] = ps.ps3000aGetValuesBulk(chandle, ctypes.byref(cmaxSamples), 0, num_buffers-1, 1, 0, ctypes.byref(overflow))
            assert_pico_ok(status["GetValuesBulk"])
            
            acq_end = time.time()
            acquisition_times.append(acq_end - acq_start)
            acquisition_count += 1
            
            if acquisition_count % 10 == 0:
                print(f"Completed {acquisition_count} acquisitions...")
        
        # Close unit
        status["closeUnit"] = ps.ps3000aCloseUnit(chandle)
        assert_pico_ok(status["closeUnit"])
        
        # Calculate results
        total_time = time.time() - start_time
        avg_acq_time = np.mean(acquisition_times)
        sampling_rate = acquisition_count / total_time
        data_throughput = (acquisition_count * samples * num_buffers) / total_time
        
        results = {
            'mode': 'Block Mode',
            'timebase': timebase,
            'samples_per_acquisition': samples,
            'num_buffers': num_buffers,
            'total_acquisitions': acquisition_count,
            'total_time': total_time,
            'avg_acquisition_time': avg_acq_time,
            'sampling_rate_hz': sampling_rate,
            'data_throughput_samples_per_sec': data_throughput,
            'latency_ms': avg_acq_time * 1000
        }
        
        self.results['block_mode'] = results
        return results
    
    def benchmark_fast_block_mode(self, timebase=0, samples=15000, num_buffers=12, duration=10):
        """Benchmark optimized block mode"""
        print(f"\n=== Benchmarking Fast Block Mode ===")
        print(f"Timebase: {timebase}, Samples: {samples}, Buffers: {num_buffers}")
        
        # This uses the same benchmark function but with optimized parameters
        results = self.benchmark_block_mode(timebase, samples, num_buffers, duration)
        results['mode'] = 'Fast Block Mode'
        return results
    
    def print_results(self):
        """Print benchmark results"""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        for mode_name, results in self.results.items():
            print(f"\n{results['mode']}:")
            print(f"  Timebase: {results['timebase']}")
            print(f"  Samples per acquisition: {results['samples_per_acquisition']}")
            print(f"  Number of buffers: {results['num_buffers']}")
            print(f"  Total acquisitions: {results['total_acquisitions']}")
            print(f"  Total time: {results['total_time']:.2f} seconds")
            print(f"  Average acquisition time: {results['avg_acquisition_time']:.4f} seconds")
            print(f"  Sampling rate: {results['sampling_rate_hz']:.2f} Hz")
            print(f"  Data throughput: {results['data_throughput_samples_per_sec']:.0f} samples/sec")
            print(f"  Latency: {results['latency_ms']:.2f} ms")
    
    def plot_results(self):
        """Plot benchmark results"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        modes = list(self.results.keys())
        sampling_rates = [self.results[mode]['sampling_rate_hz'] for mode in modes]
        latencies = [self.results[mode]['latency_ms'] for mode in modes]
        throughputs = [self.results[mode]['data_throughput_samples_per_sec'] for mode in modes]
        
        # Sampling rate comparison
        ax1.bar(modes, sampling_rates, color='skyblue')
        ax1.set_ylabel('Sampling Rate (Hz)')
        ax1.set_title('Sampling Rate Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Latency comparison
        ax2.bar(modes, latencies, color='lightcoral')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Latency Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        ax3.bar(modes, throughputs, color='lightgreen')
        ax3.set_ylabel('Data Throughput (samples/sec)')
        ax3.set_title('Data Throughput Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Performance score (higher is better)
        performance_scores = [rate / latency for rate, latency in zip(sampling_rates, latencies)]
        ax4.bar(modes, performance_scores, color='gold')
        ax4.set_ylabel('Performance Score (Hz/ms)')
        ax4.set_title('Overall Performance Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('picoscope_performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_benchmark(self, duration=30):
        """Run complete benchmark suite"""
        print("Starting comprehensive PicoScope performance benchmark...")
        print(f"Each test will run for {duration} seconds")
        
        # Test 1: Standard block mode (current implementation)
        try:
            print("\n" + "="*50)
            print("TEST 1: Standard Block Mode (Current Implementation)")
            print("="*50)
            self.benchmark_block_mode(timebase=2, samples=30000, num_buffers=20, duration=duration)
        except Exception as e:
            print(f"Error in standard block mode: {e}")
        
        # Test 2: Fast block mode with optimized parameters
        try:
            print("\n" + "="*50)
            print("TEST 2: Fast Block Mode (Optimized Parameters)")
            print("="*50)
            results = self.benchmark_fast_block_mode(timebase=0, samples=15000, num_buffers=12, duration=duration)
            self.results['fast_block_mode'] = results
        except Exception as e:
            print(f"Error in fast block mode: {e}")
        
        # Test 3: Aggressive optimization mode
        try:
            print("\n" + "="*50)
            print("TEST 3: Aggressive Optimization Mode")
            print("="*50)
            results = self.benchmark_block_mode(timebase=0, samples=10000, num_buffers=8, duration=duration)
            results['mode'] = 'Aggressive Mode'
            self.results['aggressive_mode'] = results
        except Exception as e:
            print(f"Error in aggressive mode: {e}")
        
        # Test 4: Conservative optimization mode
        try:
            print("\n" + "="*50)
            print("TEST 4: Conservative Optimization Mode")
            print("="*50)
            results = self.benchmark_block_mode(timebase=1, samples=20000, num_buffers=15, duration=duration)
            results['mode'] = 'Conservative Mode'
            self.results['conservative_mode'] = results
        except Exception as e:
            print(f"Error in conservative mode: {e}")
        
        # Print and plot results
        self.print_results()
        self.plot_results()
        
        # Generate recommendations
        self.generate_recommendations()
    
    def generate_recommendations(self):
        """Generate performance recommendations based on results"""
        print("\n" + "="*60)
        print("PERFORMANCE RECOMMENDATIONS")
        print("="*60)
        
        if not self.results:
            print("No results available for recommendations")
            return
        
        # Find best performing mode
        best_mode = max(self.results.keys(), 
                       key=lambda mode: self.results[mode]['sampling_rate_hz'])
        
        print(f"\nBest performing mode: {self.results[best_mode]['mode']}")
        print(f"Sampling rate: {self.results[best_mode]['sampling_rate_hz']:.2f} Hz")
        print(f"Latency: {self.results[best_mode]['latency_ms']:.2f} ms")
        print(f"Data throughput: {self.results[best_mode]['data_throughput_samples_per_sec']:.0f} samples/sec")
        
        # Calculate improvements
        if 'block_mode' in self.results:
            baseline = self.results['block_mode']
            print(f"\nPerformance improvements over baseline:")
            
            for mode_name, results in self.results.items():
                if mode_name != 'block_mode':
                    speed_improvement = results['sampling_rate_hz'] / baseline['sampling_rate_hz']
                    latency_improvement = baseline['latency_ms'] / results['latency_ms']
                    throughput_improvement = results['data_throughput_samples_per_sec'] / baseline['data_throughput_samples_per_sec']
                    
                    print(f"  {results['mode']}:")
                    print(f"    - Speed: {speed_improvement:.1f}x faster")
                    print(f"    - Latency: {latency_improvement:.1f}x better")
                    print(f"    - Throughput: {throughput_improvement:.1f}x higher")
        
        print("\nSpecific recommendations for your system:")
        
        # Based on current performance level
        current_rate = self.results[best_mode]['sampling_rate_hz']
        if current_rate > 100:
            print("✓ Your system has excellent performance capability")
            print("  - Consider using streaming mode for continuous acquisition")
            print("  - You can handle real-time processing applications")
        elif current_rate > 50:
            print("✓ Your system has good performance capability")
            print("  - Fast block mode should work well for your application")
            print("  - Consider reducing GUI update frequency for better performance")
        else:
            print("! Your system may need optimization")
            print("  - Start with conservative optimization mode")
            print("  - Consider hardware upgrades if real-time performance is critical")
        
        print("\nImplementation recommendations:")
        print("1. For immediate improvement, use fast block mode:")
        print("   plotter = TimeSeriesPlotter(num_buffers=12, use_fast_block=True)")
        
        print("2. For maximum performance, try streaming mode:")
        print("   plotter = TimeSeriesPlotter(num_buffers=8, use_streaming=True)")
        
        print("3. For GUI performance, enable plot downsampling:")
        print("   self.enable_plot_downsampling = True")
        
        print("4. Adjust timer frequencies:")
        print("   - Data acquisition: 2-10ms intervals")
        print("   - GUI updates: 20-50ms intervals")
        
        print("\nHardware-specific optimizations:")
        if 'fast_block_mode' in self.results:
            fast_timebase = self.results['fast_block_mode']['timebase']
            print(f"- Your PicoScope supports timebase {fast_timebase} (confirmed working)")
            if fast_timebase > 0:
                print(f"- Timebase 0 is not supported by your model")
            else:
                print(f"- Your model supports the fastest timebase")
        
        print("\nMonitoring recommendations:")
        print("- Monitor CPU usage during operation")
        print("- Watch for buffer overflows in continuous operation")
        print("- Test with your specific signal characteristics")
        print("- Measure end-to-end latency including processing time")

def main():
    """Main benchmark function"""
    print("PicoScope Performance Benchmark Tool")
    print("This will test different sampling modes and measure performance")
    print("Make sure your PicoScope is connected and no other software is using it")
    
    # Ask user for benchmark duration
    try:
        duration = int(input("\nEnter benchmark duration per test (seconds, default 10): ") or "10")
    except ValueError:
        duration = 10
    
    if duration < 5:
        print("Warning: Duration too short, using 5 seconds minimum")
        duration = 5
    elif duration > 60:
        print("Warning: Duration too long, using 60 seconds maximum")
        duration = 60
    
    benchmark = PicoScopePerformanceBenchmark()
    
    print(f"\nStarting benchmark with {duration} seconds per test...")
    print("This will take approximately {0} seconds total".format(duration * 4))
    
    # Run benchmark
    try:
        benchmark.run_full_benchmark(duration=duration)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark error: {e}")
        print("Make sure:")
        print("- PicoScope is connected")
        print("- No other software is using the PicoScope")
        print("- PicoScope drivers are installed")

if __name__ == "__main__":
    main() 