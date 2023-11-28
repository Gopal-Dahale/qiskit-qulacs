"""Qulacs Job"""

from qiskit.providers import JobError, JobStatus
from qiskit.providers import JobV1 as Job

from .backend_utils import DEFAULT_EXECUTOR, requires_submit


class QulacsJob(Job):
    """Qulacs Job"""

    def __init__(
        self,
        backend,
        job_id,
        fn,
        circuits,
        states,
        run_options=None,
        executor=None,
    ) -> None:
        super().__init__(backend, job_id)
        self._fn = fn
        self._circuits = circuits
        self._states = states
        self._run_options = run_options
        self._executor = executor or DEFAULT_EXECUTOR
        self._future = None

    def submit(self):
        """Submit the job to the backend for execution."""
        if self._future is not None:
            raise JobError("Qulacs job has already been submitted.")

        self._future = self._executor.submit(
            self._fn,
            self._circuits,
            self._states,
            self._run_options,
            self._job_id,
        )

    @requires_submit
    def result(self, timeout=None):
        """Get job result. The behavior is the same as the underlying
        concurrent Future objects,
        """
        return self._future.result(timeout=timeout)

    @requires_submit
    def cancel(self):
        """Attempt to cancel the job."""
        return self._future.cancel()

    @requires_submit
    def status(self):
        """Gets the status of the job by querying the Python's future"""
        # The order is important here
        if self._future.running():
            _status = JobStatus.RUNNING
        elif self._future.cancelled():
            _status = JobStatus.CANCELLED
        elif self._future.done():
            _status = (
                JobStatus.DONE if self._future.exception() is None else JobStatus.ERROR
            )
        else:
            _status = JobStatus.INITIALIZING
        return _status

    def backend(self):
        """Return the instance of the backend used for this job."""
        return self._backend

    def circuits(self):
        """Return the list of QuantumCircuit submitted for this job.

        Returns:
            list of QuantumCircuit: the list of QuantumCircuit submitted for this job.
        """
        return self._circuits

    def executor(self):
        """Return the executor for this job"""
        return self._executor
