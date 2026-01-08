package com.sportsbettingapp.ui

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sportsbettingapp.data.model.PredictionsResponseDto
import com.sportsbettingapp.data.model.RunStatusResponseDto
import com.sportsbettingapp.data.repository.ApiRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class PredictionsViewModel(
    private val savedStateHandle: SavedStateHandle,
    private val repository: ApiRepository = ApiRepository.create()
) : ViewModel() {
    private val _predictionsState = MutableStateFlow<UiState<PredictionsResponseDto>>(UiState.Idle)
    val predictionsState: StateFlow<UiState<PredictionsResponseDto>> = _predictionsState

    private val _statusState = MutableStateFlow<UiState<RunStatusResponseDto>>(UiState.Idle)
    val statusState: StateFlow<UiState<RunStatusResponseDto>> = _statusState

    private var pollingJob: Job? = null

    init {
        restoreStatusFromSavedState()
    }

    fun startPolling(runId: String) {
        if (pollingJob?.isActive == true && savedStateHandle[RUN_ID_KEY] == runId) {
            return
        }
        savedStateHandle[RUN_ID_KEY] = runId
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            while (true) {
                try {
                    val status = withContext(Dispatchers.IO) {
                        repository.getRunStatus(runId)
                    }
                    saveStatus(status)
                    _statusState.value = UiState.Success(status)
                    if (status.status == "done") {
                        loadPredictions(runId)
                        break
                    }
                    if (status.status == "failed") {
                        break
                    }
                } catch (exception: Exception) {
                    _statusState.value = UiState.Error(exception.localizedMessage ?: "Failed to load status")
                    break
                }
                delay(2_500L)
            }
        }
    }

    private fun loadPredictions(runId: String) {
        _predictionsState.value = UiState.Loading
        viewModelScope.launch {
            try {
                val predictions = withContext(Dispatchers.IO) {
                    repository.getPredictions(runId)
                }
                _predictionsState.value = UiState.Success(predictions)
            } catch (e: Exception) {
                _predictionsState.value = UiState.Error(e.message ?: "Failed to load predictions")
            }
        }
    }

    companion object {
        private const val RUN_ID_KEY = "run_id"
        private const val STATUS_KEY = "run_status"
        private const val PROGRESS_KEY = "run_progress"
        private const val MESSAGE_KEY = "run_message"
        private const val ERROR_KEY = "run_error"
        private const val PREDICTIONS_COUNT_KEY = "run_predictions_count"
        private const val TRACKED_BETS_COUNT_KEY = "run_tracked_bets_count"
    }

    private fun saveStatus(status: RunStatusResponseDto) {
        savedStateHandle[STATUS_KEY] = status.status
        savedStateHandle[PROGRESS_KEY] = status.progressPercent
        savedStateHandle[MESSAGE_KEY] = status.message
        savedStateHandle[ERROR_KEY] = status.error
        savedStateHandle[PREDICTIONS_COUNT_KEY] = status.predictionsCount
        savedStateHandle[TRACKED_BETS_COUNT_KEY] = status.trackedBetsCount
    }

    private fun restoreStatusFromSavedState() {
        val status = savedStateHandle.get<String>(STATUS_KEY) ?: return
        val progress = savedStateHandle.get<Int>(PROGRESS_KEY) ?: 0
        val predictionsCount = savedStateHandle.get<Int>(PREDICTIONS_COUNT_KEY) ?: 0
        val trackedBetsCount = savedStateHandle.get<Int>(TRACKED_BETS_COUNT_KEY) ?: 0
        val message = savedStateHandle.get<String>(MESSAGE_KEY)
        val error = savedStateHandle.get<String>(ERROR_KEY)
        _statusState.value = UiState.Success(
            RunStatusResponseDto(
                id = savedStateHandle.get<String>(RUN_ID_KEY) ?: "",
                status = status,
                progressPercent = progress,
                message = message,
                predictionsCount = predictionsCount,
                trackedBetsCount = trackedBetsCount,
                error = error
            )
        )
    }
}
