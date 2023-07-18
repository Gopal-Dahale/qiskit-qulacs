from typing import Optional
from qiskit.providers import JobStatus, JobV1
from qiskit.providers.backend import Backend
from typing import List, Optional, Union
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

def _get_result(tasks):

	expt_results = []
	for qulacs_state, circuit in tasks:
		circuit.update_quantum_state(qulacs_state)
		final_state = qulacs_state.get_vector()
		data = ExperimentResultData(
                    statevector=final_state,
                )
		expt_results.append(ExperimentResult(
			shots=0,
            success=True,
            status=JobStatus.DONE,
            data=data,
        ))

	return expt_results

class QulacsJob(JobV1):
	def __init__(self, task_id, tasks, backend, **metadata: Optional[dict]) -> None:
		super().__init__(backend=backend, job_id=task_id, metadata=metadata)
		self._task_id = task_id
		self._backend = backend
		self._metadata = metadata
		self._tasks = tasks

	def submit(self):
		return

	def task_id(self) -> str:
		"""Return a unique id identifying the task."""
		return self._task_id

	def result(self) -> Result:
		experiment_results = _get_result(tasks=self._tasks)
		return Result(
			backend_name=self._backend,
			backend_version=self._backend.version,
			job_id=self._task_id,
			qobj_id=0,
			success=True,
			results=experiment_results,
			status=self.status())

	def status(self):
		return JobStatus.DONE

