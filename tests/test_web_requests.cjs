const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function main() {
    let calls = 0, lastSignal;
    const context = vm.createContext({AbortController, setTimeout, clearTimeout});
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/request.js'), 'utf8'), context);
    context.fetch = async (url, options) => {
        calls++;
        lastSignal = options.signal;
        assert.equal(options.method, 'POST');
        return {ok: true, status: 200, json: async () => ({done: true})};
    };
    assert.equal((await context.api('/api/index/start', {method: 'POST'}, 20)).done, true);
    await new Promise(resolve => setTimeout(resolve, 30));
    assert.equal(lastSignal.aborted, false, 'successful request timer must be cleared');
    assert.equal(calls, 1, 'writes must not be retried automatically');
    context.fetch = async () => ({ok: false, status: 409, json: async () => ({error: 'already running'})});
    await assert.rejects(context.api('/api/index/start'), /already running/);
    context.fetch = async () => ({ok: false, status: 502, json: async () => {throw Error('invalid JSON');}});
    await assert.rejects(context.api('/api/status'), /HTTP 502/);
    for (const slowBody of [false, true]) {
        calls = 0;
        context.fetch = async (url, options) => {
            calls++;
            const pending = () => new Promise((resolve, reject) => {
                options.signal.addEventListener('abort', () => reject(Error('aborted')), {once: true});
            });
            return slowBody ? {ok: true, json: pending} : pending();
        };
        await assert.rejects(context.api('/api/ask', {method: 'POST'}, 5), /서버 작업이 취소된 것은 아닙니다/);
        assert.equal(calls, 1);
    }
    // Run the actual UI wrapper: failed requests must restore buttons and retain an actionable notice.
    const buttons = [{disabled: false}, {disabled: false}];
    const notice = {textContent: ''};
    context.document = {querySelectorAll: () => buttons};
    context.notice = notice;
    context.indexControls = () => {};
    const html = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
    assert.ok(html.includes('<script src="/request.js"></script>'));
    const wrapper = html.split('\n').find(line => line.startsWith('async function run('));
    vm.runInContext(wrapper, context);
    await context.run('waiting', () => context.api('/api/ask', {}, 5));
    assert.ok(buttons.every(button => !button.disabled));
    assert.match(notice.textContent, /응답 대기 시간이 초과/);
    console.log('WEB REQUEST TESTS PASSED');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
