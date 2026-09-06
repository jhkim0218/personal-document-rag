async function api(path, options = {}, timeoutMs = 120000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(path, {...options, signal: controller.signal});
        let body;
        try {
            body = await response.json();
        } catch (error) {
            if (controller.signal.aborted) throw error;
            throw Error(`서버 응답을 읽지 못했습니다 (HTTP ${response.status}). 서버 상태를 확인하세요.`);
        }
        if (!response.ok) throw Error(body?.error || `요청 실패 (HTTP ${response.status})`);
        return body;
    } catch (error) {
        if (controller.signal.aborted) {
            throw Error('응답 대기 시간이 초과됐습니다. 서버 작업이 취소된 것은 아닙니다. 색인·설정 요청은 상태를 확인한 뒤 재시도하세요.');
        }
        throw error;
    } finally {
        clearTimeout(timer);
    }
}
